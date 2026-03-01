"""Semantic mapping of questions to wellness dimensions using embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .embedding import load_embedding_model, embed_texts
from .similarity import cosine_similarity_matrix
from .selection import select_by_margin


# -------------------------------------------------
# Dimension Set
# -------------------------------------------------

@dataclass
class DimensionSet:
    """Container for dimension definitions and their embeddings.

    Attributes:
        definitions: List of dimension definitions (e.g., "Emotional: ...").
        model_name: Name of the LLM that generated the definitions.
        names: List of dimension names (e.g., ["Emotional", "Physical", ...]).
        embeddings: Embeddings array of shape (M, D).
    """
    definitions: List[str]
    model_name: str
    names: List[str]
    embeddings: np.ndarray


# -------------------------------------------------
# Semantic Mapper
# -------------------------------------------------

class SemanticMapper:
    """Map survey questions to wellness dimensions using semantic similarity.

    This class provides a stateful wrapper for research workflows. It caches
    question embeddings and allows swapping dimension sets for comparing
    different LLM-generated definitions.

    Attributes:
        model: The sentence transformer model for embeddings.
        dimset: Current dimension set (DimensionSet or None).
        questions_df: DataFrame containing questions.
        question_embeddings: Cached embeddings for questions.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model: SentenceTransformer = load_embedding_model(embedding_model)

        self.dimset: Optional[DimensionSet] = None

        self.questions_df: Optional[pd.DataFrame] = None
        self.text_col: str = "text"
        self.qid_col: str = "qid"
        self.dataset_col: str = "dataset"
        self.question_embeddings: Optional[np.ndarray] = None

    # ---------- Dimensions ----------
    def set_dimensions(self, definitions: List[str], dimension_model_name: str) -> None:
        """Set dimension definitions and generate embeddings.

        Args:
            definitions: List of dimension definitions with format "Name: definition".
            dimension_model_name: Name of the LLM that generated these definitions.
        """
        names = [d.split(":", 1)[0].strip() for d in definitions]
        emb = embed_texts(self.model, definitions)
        self.dimset = DimensionSet(
            definitions=definitions,
            model_name=dimension_model_name,
            names=names,
            embeddings=emb,
        )

    # ---------- Questions ----------
    def set_questions_df(
        self,
        df: pd.DataFrame,
        *,
        text_col: str = "text",
        qid_col: str = "qid",
        dataset_col: str = "dataset",
    ) -> None:
        """Set questions DataFrame and generate embeddings.

        Args:
            df: DataFrame containing survey questions.
            text_col: Column containing question text.
            qid_col: Column containing question identifiers.
            dataset_col: Column containing dataset/source names.

        Raises:
            ValueError: If text_col is not in the DataFrame.
        """
        if text_col not in df.columns:
            raise ValueError(f"Missing column: {text_col}")

        self.text_col = text_col
        self.qid_col = qid_col
        self.dataset_col = dataset_col

        self.questions_df = df.copy()
        texts = self.questions_df[text_col].astype(str).tolist()
        self.question_embeddings = embed_texts(self.model, texts)

    # ---------- Mapping ----------
    def map_questions_to_dimensions(self, *, delta: float) -> pd.DataFrame:
        """Map questions to dimensions based on cosine similarity.

        Uses the margin-based selection to assign questions to one or more
        dimensions based on similarity scores.

        Args:
            delta: Margin threshold. A question is assigned to dimensions
                within delta of the maximum similarity score.

        Returns:
            pd.DataFrame with columns:
                - qid: Question identifier
                - dataset: Source dataset
                - text: Question text
                - dimension_model: LLM that generated dimension definitions
                - dimensions: List of assigned dimension names
                - scores: List of similarity scores for each assigned dimension

        Raises:
            ValueError: If dimensions or questions have not been set.
        """
        if self.dimset is None:
            raise ValueError("Dimensions not set. Call set_dimensions(...) first.")
        if self.questions_df is None or self.question_embeddings is None:
            raise ValueError("Questions not set. Call set_questions_df(...) first.")

        sim = cosine_similarity_matrix(self.question_embeddings, self.dimset.embeddings)

        dims_out: List[List[str]] = []
        scores_out: List[List[float]] = []

        for i in range(len(self.questions_df)):
            row_scores = sim[i]
            idx = select_by_margin(row_scores, delta)

            dims_out.append([self.dimset.names[j] for j in idx])
            scores_out.append([float(row_scores[j]) for j in idx])

        out = self.questions_df.copy()
        out["dimension_model"] = self.dimset.model_name
        out["dimensions"] = dims_out
        out["scores"] = scores_out

        keep_cols = [
            c for c in [self.qid_col, self.dataset_col, self.text_col, "dimension_model"]
            if c in out.columns
        ]
        keep_cols += ["dimensions", "scores"]
        return out[keep_cols]
    
    def extract_top1_from_mapped(
        self,
        mapped_df: pd.DataFrame,
        *,
        qid_col: str = "qid",
        text_col: str = "text",
        dimensions_col: str = "dimensions",
        source_col_in: str = "dimension_model",
        source_out_col: str = "source",
    ) -> pd.DataFrame:
        """Convert multi-label mapping to single-label (top-1) format.

        Args:
            mapped_df: DataFrame with multi-label mappings.
            qid_col: Column for question IDs.
            text_col: Column for question text.
            dimensions_col: Column containing list of assigned dimensions.
            source_col_in: Column containing the source model name.
            source_out_col: Output column name for source.

        Returns:
            pd.DataFrame with columns:
                - qid: Question identifier
                - text: Question text
                - answer: First (top) dimension
                - source: Source model name
        """

        df = mapped_df.copy()

        # take first dimension as top1
        df["answer"] = df[dimensions_col].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
        )

        df[source_out_col] = df[source_col_in]

        return (
            df[[qid_col, text_col, "answer", source_out_col]]
            .dropna(subset=["answer"])
            .reset_index(drop=True)
        )

    def map_to_llm_dimension_definitions(
        self,
        dimension_sets: dict,
        *,
        deltas: List[float],
    ) -> pd.DataFrame:
        """
        Map questions using multiple dimension definition sets and delta thresholds.

        Args:
            dimension_sets: Dict[str, List[str]]
                Mapping from model_name -> list of dimension definitions.
            deltas: List of margin thresholds.

        Returns:
            pd.DataFrame
                Merged mapping results across all models and deltas.
        """

        if self.questions_df is None:
            raise ValueError("Questions not set. Call set_questions_df(...) first.")

        all_results = []

        for model_name, dim_defs in dimension_sets.items():
            print(f"\n=== Mapping using {model_name} ===")

            # set dimension definitions
            self.set_dimensions(dim_defs, dimension_model_name=model_name)

            for delta in deltas:
                mapped = self.map_questions_to_dimensions(delta=delta).copy()
                mapped["delta"] = float(delta)

                all_results.append(mapped)

        if not all_results:
            raise ValueError("No mapping results generated.")

        merged = pd.concat(all_results, ignore_index=True)

        return merged