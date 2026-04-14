"""Semantic scoring of questions to wellness dimensions using embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# -------------------------------------------------
# Dimension Set
# -------------------------------------------------

@dataclass
class DimensionSet:
    """Container for one model's dimension definitions and embeddings."""
    definitions: List[str]
    model_name: str
    names: List[str]
    embeddings: np.ndarray


# -------------------------------------------------
# Semantic Mapper
# -------------------------------------------------

class SemanticMapper:
    """
    Score survey questions against wellness dimensions using cosine similarity.

    Main workflow:
        1. set_questions_df(...)
        2. score_questions_to_dimensions()
        3. mean_scores_across_models(...)
        4. score_and_average(...)
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model: SentenceTransformer = SentenceTransformer(embedding_model)

        # store multiple dimension sets
        self.dimsets: dict[str, DimensionSet] = {}

        self.questions_df: Optional[pd.DataFrame] = None
        self.text_col: str = "text"
        self.qid_col: str = "qid"
        self.dataset_col: str = "dataset"
        self.question_embeddings: Optional[np.ndarray] = None

    # ---------- Internal helpers ----------
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def _cosine_similarity_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between row vectors of two matrices.

        Args:
            A: shape (N, D)
            B: shape (M, D)

        Returns:
            similarity matrix of shape (N, M)
        """
        eps = 1e-12
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
        B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
        return A_norm @ B_norm.T

    # ---------- Dimensions ----------
    def set_dimensions(self, dimension_sets: dict[str, list[str]]) -> None:
        """
        Set multiple dimension definition sets at once.

        Args:
            dimension_sets:
                dict of {model_name: [definitions]}
                where each definition is like "Emotional: ..."
        """
        self.dimsets = {}

        for model_name, definitions in dimension_sets.items():
            names = [d.split(":", 1)[0].strip() for d in definitions]
            emb = self._embed_texts(definitions)

            self.dimsets[model_name] = DimensionSet(
                definitions=definitions,
                model_name=model_name,
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
        """
        Set questions DataFrame and compute question embeddings.
        """
        if text_col not in df.columns:
            raise ValueError(f"Missing column: {text_col}")

        self.text_col = text_col
        self.qid_col = qid_col
        self.dataset_col = dataset_col

        self.questions_df = df.copy().reset_index(drop=True)
        texts = self.questions_df[text_col].astype(str).tolist()
        self.question_embeddings = self._embed_texts(texts)

    # ---------- Core scoring ----------
    def score_questions_to_dimensions(self) -> pd.DataFrame:
        """
        Score questions against all loaded dimension sets.

        Returns:
            Long-format dataframe:
                qid / text / dataset / dimension_model / 8 dimension score columns
        """
        if not self.dimsets:
            raise ValueError("Dimensions not set. Call set_dimensions(...) first.")
        if self.questions_df is None or self.question_embeddings is None:
            raise ValueError("Questions not set. Call set_questions_df(...) first.")

        all_results = []

        for model_name, dimset in self.dimsets.items():
            sim = self._cosine_similarity_matrix(
                self.question_embeddings,
                dimset.embeddings,
            )  # shape: (N_questions, 8)

            out = self.questions_df.copy()
            out["dimension_model"] = model_name

            for j, dim_name in enumerate(dimset.names):
                out[dim_name] = sim[:, j]

            keep_cols = [
                c for c in [self.qid_col, self.dataset_col, self.text_col, "dimension_model"]
                if c in out.columns
            ] + dimset.names

            all_results.append(out[keep_cols])

        return pd.concat(all_results, ignore_index=True)

    def mean_scores_across_models(
        self,
        scored_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Average dimension scores across all models.

        Args:
            scored_df:
                output of score_questions_to_dimensions().
                If None, will compute internally.

        Returns:
            Wide dataframe:
                qid / text / dataset / 8 dimension mean score columns
        """
        if scored_df is None:
            scored_df = self.score_questions_to_dimensions()

        meta_cols = {self.qid_col, self.text_col, self.dataset_col, "dimension_model"}
        dim_cols = [c for c in scored_df.columns if c not in meta_cols]

        mean_df = (
            scored_df.groupby(self.qid_col, as_index=False)[dim_cols]
            .mean()
        )

        extra_cols = [c for c in [self.text_col, self.dataset_col] if c in self.questions_df.columns]

        if extra_cols:
            meta_df = self.questions_df[[self.qid_col] + extra_cols].drop_duplicates(subset=[self.qid_col])
            mean_df = mean_df.merge(meta_df, on=self.qid_col, how="left")

            ordered_cols = [self.qid_col] + extra_cols + dim_cols
            mean_df = mean_df[ordered_cols]

        return mean_df

    def score_and_average(
        self,
        dimension_sets: dict[str, list[str]],
    ) -> pd.DataFrame:
        """
        One-call convenience method.

        Args:
            dimension_sets:
                dict of {model_name: [definitions]}

        Returns:
            Mean score dataframe across all models.
        """
        self.set_dimensions(dimension_sets)
        scored_df = self.score_questions_to_dimensions()
        mean_df = self.mean_scores_across_models(scored_df)
        return mean_df
