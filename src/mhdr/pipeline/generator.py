from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional
import gc
import re

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


@dataclass
class GenerationResult:
    target_dims: List[str]
    prompt: str
    outputs: List[str]


class SeededQuestionGenerator:
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _clear_memory(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _format_target_dims(self, target_dims: Sequence[str]) -> str:
        target_dims = list(target_dims)
        if len(target_dims) == 1:
            return target_dims[0]
        if len(target_dims) == 2:
            return f"{target_dims[0]} and {target_dims[1]}"
        return ", ".join(target_dims[:-1]) + f", and {target_dims[-1]}"

    def _build_prompt(
        self,
        target_dims: Sequence[str],
        seed_texts: Sequence[str],
    ) -> str:
        target_str = self._format_target_dims(target_dims)
        seed_block = "\n".join(f"- {s}" for s in seed_texts)

        return f"""You are generating high-quality self-report mental health survey items.

        Task:
        Create ONE new self-report statement that reflects the following dimensions:
        {target_str}

        STRICT REQUIREMENTS:
        - Must be ONE clear declarative sentence (NOT a question)
        - Must be concise, specific, and easy to understand
        - Must use clear subject reference (avoid ambiguous or unclear pronouns)
        - Must describe personal feelings, behaviors, or tendencies
        - Must sound like a psychological self-report item
        - Must NOT copy or closely rephrase any example
        - Must NOT include contradictions, vague wording, or nonsensical phrases

        STYLE GUIDELINES:
        - Prefer simple and direct sentence structure
        - Avoid unnecessary complexity or overly abstract wording
        - Avoid phrases like:
        - "self-report"
        - "I have a strong immune system"
        - overly generic statements like "I am good"
        - Avoid repeating the same structure as examples
        - Aim for diversity in wording and meaning

        GOOD EXAMPLE STYLE:
        - "I am able to manage my emotions effectively during stressful situations."
        - "I seek support from others when I feel overwhelmed."

        BAD EXAMPLES (DO NOT DO):
        - Questions (e.g., "How often do you feel sad?")
        - Rewriting the same sentence structure
        - Vague or unclear statements
        - Ambiguous pronouns (e.g., "this", "that", "it" without clear meaning)
        - Contradictory statements

        Original examples:
        {seed_block}

        New statement:
        """

    def _clean_output(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"^\s*[-•0-9.)]+\s*", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return text

        parts = re.split(r"(?<=[.!?])\s+", text)
        return parts[0].strip()

    def _tokenize_words(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _is_valid_length(
        self,
        text: str,
        *,
        min_words: int = 12,
    ) -> bool:
        return len(self._tokenize_words(text)) >= min_words

    def _get_ngrams(self, text: str, n: int = 3) -> set[tuple[str, ...]]:
        tokens = self._tokenize_words(text)
        if len(tokens) < n:
            return set()
        return {
            tuple(tokens[i:i + n])
            for i in range(len(tokens) - n + 1)
        }

    def _ngram_jaccard(self, a: str, b: str, n: int = 3) -> float:
        A = self._get_ngrams(a, n=n)
        B = self._get_ngrams(b, n=n)

        if not A or not B:
            return 0.0

        return len(A & B) / len(A | B)

    def _is_too_similar(
        self,
        text: str,
        reference_texts: Sequence[str],
        *,
        ngram_n: int = 3,
        jaccard_threshold: float = 0.5,
    ) -> bool:
        for ref in reference_texts:
            if self._ngram_jaccard(text, ref, n=ngram_n) >= jaccard_threshold:
                return True
        return False

    def _accept_candidate(
        self,
        text: str,
        *,
        base_references: Sequence[str],
        collected: Sequence[str],
        min_words: int = 12,
        ngram_n: int = 3,
        jaccard_threshold: float = 0.5,
    ) -> bool:
        if not text.strip():
            return False

        if not self._is_valid_length(text, min_words=min_words):
            return False

        if self._is_too_similar(
            text,
            reference_texts=base_references,
            ngram_n=ngram_n,
            jaccard_threshold=jaccard_threshold,
        ):
            return False

        if self._is_too_similar(
            text,
            reference_texts=collected,
            ngram_n=ngram_n,
            jaccard_threshold=jaccard_threshold,
        ):
            return False

        return True

    def generate(
        self,
        target_dims: Sequence[str],
        seed_texts: Sequence[str],
        *,
        batch_size: int = 20,
        max_new_tokens: int = 32,
        temperature: float = 0.9,
        top_p: float = 0.95,
    ) -> List[str]:
        """
        Generate one raw batch only.
        No filtering, no retry logic here.
        """
        prompt = self._build_prompt(target_dims, seed_texts)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=batch_size,
            )

        decoded = [
            self._clean_output(self.tokenizer.decode(o, skip_special_tokens=True))
            for o in outputs
        ]

        del outputs
        del inputs
        self._clear_memory()

        return decoded

    def collect(
        self,
        target_dims: Sequence[str],
        seed_texts: Sequence[str],
        *,
        n_questions: int = 30,
        existing_texts: Optional[Sequence[str]] = None,
        min_words: int = 12,
        ngram_n: int = 3,
        jaccard_threshold: float = 0.5,
        batch_size: int = 20,
        max_new_tokens: int = 32,
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_total_batches: int = 50,
        require_exact_count: bool = True,
    ) -> GenerationResult:
        """
        Collector:
        repeatedly calls generate() until enough valid outputs are collected.
        """
        target_dims = sorted(target_dims)
        prompt = self._build_prompt(target_dims, seed_texts)

        if existing_texts is None:
            existing_texts = []

        collected: List[str] = []
        seen = set()

        base_references = list(seed_texts) + list(existing_texts)

        batch_idx = 0
        while len(collected) < n_questions:
            batch_idx += 1

            if batch_idx > max_total_batches:
                if require_exact_count:
                    raise RuntimeError(
                        f"Could not collect {n_questions} valid outputs for {target_dims}. "
                        f"Collected {len(collected)} after {max_total_batches} batches."
                    )
                break

            raw_outputs = self.generate(
                target_dims=target_dims,
                seed_texts=seed_texts,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            for text in raw_outputs:
                key = text.lower().strip()

                if not key or key in seen:
                    continue

                if not self._accept_candidate(
                    text,
                    base_references=base_references,
                    collected=collected,
                    min_words=min_words,
                    ngram_n=ngram_n,
                    jaccard_threshold=jaccard_threshold,
                ):
                    continue

                seen.add(key)
                collected.append(text)

                if len(collected) >= n_questions:
                    break

            del raw_outputs
            self._clear_memory()

        return GenerationResult(
            target_dims=list(target_dims),
            prompt=prompt,
            outputs=collected[:n_questions],
        )

    def generate_to_df(
        self,
        target_dims: Sequence[str],
        seed_texts: Sequence[str],
        *,
        n_questions: int = 30,
        existing_texts: Optional[Sequence[str]] = None,
        min_words: int = 12,
        ngram_n: int = 3,
        jaccard_threshold: float = 0.5,
        batch_size: int = 20,
        max_new_tokens: int = 32,
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_total_batches: int = 50,
        require_exact_count: bool = True,
    ) -> pd.DataFrame:
        result = self.collect(
            target_dims=target_dims,
            seed_texts=seed_texts,
            n_questions=n_questions,
            existing_texts=existing_texts,
            min_words=min_words,
            ngram_n=ngram_n,
            jaccard_threshold=jaccard_threshold,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            max_total_batches=max_total_batches,
            require_exact_count=require_exact_count,
        )

        return pd.DataFrame({
            "target_dims": [", ".join(result.target_dims)] * len(result.outputs),
            "generated_text": result.outputs,
        })