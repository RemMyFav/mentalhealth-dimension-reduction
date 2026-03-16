from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
from typing import List
import re

@dataclass
class GenerationRecord:
    qid: str
    text: str
    seeds: List[str]
    status: str  # "seed" | "kept" | "rejected"
    reject_reason: Optional[str] = None


class GenerationCSVStore:
    COLUMNS = ["qid", "text", "seeds", "status", "reject_reason"]

    def __init__(self, csv_path: Path, questions: Optional[Path] = None):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        # only initialize if file doesn't exist
        if not self.csv_path.exists():
            self._initialize(questions_path=questions)

    def _initialize(self, questions_path: Optional[Path] = None):
        df = pd.DataFrame(columns=self.COLUMNS)

        if questions_path:
            survey_df = pd.read_csv(questions_path)

            if "qid" not in survey_df.columns or "text" not in survey_df.columns:
                raise ValueError("questions CSV must contain 'qid' and 'text' columns")

            seed_df = pd.DataFrame({
                "qid": survey_df["qid"].astype(str),
                "text": survey_df["text"].astype(str),
                "seeds": ["[]"] * len(survey_df),      # JSON list
                "status": ["seed"] * len(survey_df),
                "reject_reason": [""] * len(survey_df),
            })

            df = pd.concat([df, seed_df], ignore_index=True)

        df.to_csv(self.csv_path, index=False)

    def add_record(self, record: GenerationRecord, *, allow_duplicate_qid: bool = False):
        df = pd.read_csv(self.csv_path)

        if (not allow_duplicate_qid) and (df["qid"].astype(str) == str(record.qid)).any():
            return  # or raise ValueError(f"qid already exists: {record.qid}")

        new_row = {
            "qid": str(record.qid),
            "text": str(record.text),
            "seeds": json.dumps(record.seeds, ensure_ascii=False),  # safer than join
            "status": record.status,
            "reject_reason": record.reject_reason or "",
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(self.csv_path, index=False)
    
    def load_df(self) -> pd.DataFrame:
        return pd.read_csv(self.csv_path)
    
    def sample_seeds(
        self,
        k: int,
        seed_ratio: float = 0.7,
    ) -> List[str]:

        df = self.load_df()

        seed_df = df[df["status"] == "seed"]
        kept_df = df[df["status"] == "kept"]

        n_seed = int(round(k * seed_ratio))
        n_kept = k - n_seed

        sampled = []

        if len(seed_df) > 0:
            sampled += seed_df.sample(
                n=min(n_seed, len(seed_df)),
                replace=False
            )["text"].astype(str).tolist()

        if len(kept_df) > 0:
            sampled += kept_df.sample(
                n=min(n_kept, len(kept_df)),
                replace=False
            )["text"].astype(str).tolist()

        return sampled


class SeedCentroidScorer:
    def __init__(self, seed_df: pd.DataFrame, text_col: str = "text"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.seed_texts = seed_df[text_col].astype(str).tolist()

        seed_emb = self.model.encode(
            self.seed_texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        seed_emb = self._normalize(seed_emb)

        centroid = np.mean(seed_emb, axis=0, keepdims=True)

        self.centroid = self._normalize(centroid)

    def _normalize(self, x: np.ndarray):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def score(self, text: str) -> float:
        emb = self.model.encode([text], convert_to_numpy=True)
        emb = self._normalize(emb)

        sim_mat = emb @ self.centroid.T          # shape (1,1)
        return float(sim_mat.item())             # scalar

   
class FlanGenerator:
    def __init__(self, model_name="google/flan-t5-base"):  
        self.device = "cpu"  

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 30,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode(): 
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
            )

        results = [
            self.tokenizer.decode(o, skip_special_tokens=True)
            for o in outputs
        ]

        del inputs
        del outputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return results
 
class GeneratorLoop:
    def __init__(
        self,
        gen: FlanGenerator,
        store: GenerationCSVStore,
        scorer: SeedCentroidScorer,
    ):
        self.gen = gen
        self.store = store
        self.scorer = scorer

    def _build_prompt(self, seeds: List[str]) -> str:
        seeds_text = "\n".join([f"- {s}" for s in seeds])
        return f"""
        Based on the following examples, create a NEW self-report mental health statement.

        Requirements:
        - One clear declarative sentence
        - Natural and realistic
        - Do not copy phrases from the originals

        Original examples:
        {seeds_text}
        """

    def _norm(self, s: str) -> str:
        # normalize for dedup (case/space/punctuation-light)
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s

    def run(
    self,
    rounds: int = 5,
    candidates_per_round: int = 100,
    keep_ratio: float = 0.3,
    seed_k: int = 5,
    ):
        df_db = self.store.load_df()
        existing_norms = set(self._norm(t) for t in df_db["text"].astype(str).tolist())

        run_id = 0
         # counter within this run()

        for step in range(rounds):
            j = 0 
            seeds = self.store.sample_seeds(seed_k)
            prompt = self._build_prompt(seeds)

            results = self.gen.generate(
                prompt,
                num_return_sequences=candidates_per_round,
                max_new_tokens=30,
                temperature=0.95,
                top_p=0.95,
            )

            scored = []
            seen_batch = set()

            for text in results:
                text = str(text).strip()
                if not text:
                    continue

                n = self._norm(text)
                if len(text.split()) < 10:
                    continue
                if n in seen_batch:
                    continue
                if n in existing_norms:
                    continue

                sim = self.scorer.score(text)
                scored.append((text, sim))
                seen_batch.add(n)

            if not scored:
                print(f"Round {step} | Generated: 0 | Kept: 0")
                continue

            scored.sort(key=lambda x: x[1], reverse=True)
            n_keep = max(1, int(len(scored) * keep_ratio))
            top_candidates = scored[:n_keep]

            for text, sim in top_candidates:
                qid = f"gen_{run_id}_{j}"
                j += 1

                record = GenerationRecord(
                    qid=qid,
                    text=text,
                    seeds=seeds,
                    status="kept",
                )
                self.store.add_record(record)

                existing_norms.add(self._norm(text))
            run_id += 1
            print(
                f"Round {step} | Generated(after dedup): {len(scored)} | "
                f"Kept: {len(top_candidates)} | "
                f"Top sim: {top_candidates[0][1]:.3f}"
            )
