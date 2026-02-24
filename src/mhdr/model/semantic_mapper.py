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


@dataclass
class DimensionSet:
    definitions: List[str]          # e.g. ["Emotional: ...", "Physical: ..."]
    model_name: str                 # e.g. "ChatGPT-5.2"
    names: List[str]                # e.g. ["Emotional", "Physical", ...]
    embeddings: np.ndarray          # shape (M, D)


class SemanticMapper:
    """
    Thin, stateful wrapper for research workflows:
      - cache question embeddings
      - swap dimension sets
      - run mapping for many deltas
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
    
    def extract_top1_from_mapped(self,
        mapped_df: pd.DataFrame,
        *,
        qid_col: str = "qid",
        text_col: str = "text",
        dimensions_col: str = "dimensions",
        source_col_in: str = "dimension_model",
        source_out_col: str = "source",
    ) -> pd.DataFrame:
        """
        Convert multi-label mapping output to single-label (top1) format.

        Output columns:
            qid, text, answer, source
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