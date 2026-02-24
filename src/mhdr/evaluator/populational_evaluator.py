"""Population-wise disagreement evaluation for human vs LLM labels."""
import numpy as np
import pandas as pd


# -------------------------------------------------
# Populational Evaluator
# -------------------------------------------------

class PopulationalEvaluator:
    """Evaluate population-wise label disagreement across sources.

    For each question (qid), this evaluator examines labels provided by
    different sources (humans or LLM models). It computes normalized
    entropy per question and supports downsampling sources to a fixed
    number k for repeated evaluation.

    Args:
        human_df: DataFrame with human labels.
        llm_df: DataFrame with LLM labels.

    Raises:
        ValueError: If there are no overlapping qids between human and LLM data.
    """

    def __init__(self, human_df: pd.DataFrame, llm_df: pd.DataFrame):
        self.human_df = human_df.copy()
        self.llm_df = llm_df.copy()

        self.overlap_qids = self._get_overlap_qids()
        if len(self.overlap_qids) == 0:
            raise ValueError("No overlapping qids found between human and LLM data.")

    # ---------- overlap ----------
    def _get_overlap_qids(self) -> list[str]:
        """Get question IDs present in both human and LLM datasets."""
        return sorted(set(self.human_df["qid"]) & set(self.llm_df["qid"]))

    # ---------- entropy ----------
    @staticmethod
    def _entropy_norm(counts: np.ndarray) -> float:
        """Compute normalized entropy (0 to 1) for label counts.

        Args:
            counts: Array of label counts for a single question.

        Returns:
            Normalized entropy value between 0 and 1.
        """
        counts = np.asarray(counts, dtype=float)
        s = counts.sum()
        if s <= 0:
            return np.nan

        p = counts / s
        # entropy with safe mask
        p_nz = p[p > 0]
        h = -np.sum(p_nz * np.log2(p_nz))

        K = len(counts)  # this is len(label_space)
        h_max = np.log2(K) if K > 1 else 0.0
        return float(h / h_max) if h_max > 0 else 0.0

    # ---------- downsample ----------
    @staticmethod
    def _downsample_sources_once(
        df: pd.DataFrame,
        k_sources: int,
        *,
        source_col: str,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Randomly sample k_sources from available sources.

        Args:
            df: Input DataFrame.
            k_sources: Number of sources to sample.
            source_col: Column name for source identifiers.
            seed: Random seed for reproducibility.

        Returns:
            DataFrame with only the sampled sources.
        """
        rng = np.random.default_rng(seed)
        sources = df[source_col].dropna().astype(str).unique().tolist()

        if not sources:
            raise ValueError("No sources found.")

        k = min(k_sources, len(sources))
        picked = rng.choice(sources, size=k, replace=False)

        return df[df[source_col].isin(picked)].copy()

    # ---------- per-qid disagreement ----------
    def disagreement_per_qid(
        self,
        df: pd.DataFrame,
        *,
        qid_col: str = "qid",
        source_col: str = "source",
        label_col: str = "answer",
        label_space: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compute disagreement (entropy) per question.

        Args:
            df: DataFrame with labels.
            qid_col: Column for question IDs.
            source_col: Column for source identifiers.
            label_col: Column for labels/answers.
            label_space: Optional list of all possible labels.

        Returns:
            pd.DataFrame with columns:
                - qid: Question identifier
                - n_sources: Number of sources for this question
                - entropy: Normalized entropy (0=agree, 1=disagree)
        """

        x = df[[qid_col, source_col, label_col]].dropna().copy()
        x[qid_col] = x[qid_col].astype(str)
        x[source_col] = x[source_col].astype(str)
        x[label_col] = x[label_col].astype(str)

        # enforce one label per (source, qid)
        x = x.drop_duplicates(subset=[source_col, qid_col], keep="first")

        if label_space is None:
            label_space = sorted(x[label_col].unique().tolist())

        label2idx = {lab: i for i, lab in enumerate(label_space)}

        rows = []
        for qid, g in x.groupby(qid_col, sort=False):

            counts = np.zeros(len(label_space), dtype=float)

            for lab in g[label_col]:
                idx = label2idx.get(lab)
                if idx is not None:
                    counts[idx] += 1

            if counts.sum() == 0:
                continue

            score = self._entropy_norm(counts)

            rows.append({
                qid_col: qid,
                "n_sources": int(g[source_col].nunique()),
                "entropy": float(score),
            })

        return pd.DataFrame(rows)

    # ---------- repeated runs ----------
    def disagreement_many(
        self,
        which: str,
        *,
        k_sources: int = 5,
        times: int = 100,
        seed: int = 42,
        qid_col: str = "qid",
        source_col: str = "source",
        label_col: str = "answer",
        label_space: list[str] | None = None,
        restrict_to_overlap_qids: bool = True,
    ):
        """Run repeated disagreement evaluation with source downsampling.

        Each run samples k_sources and computes per-question entropy across
        them. Returns per-run results, summary statistics, and per-question
        results across all runs.

        Args:
            which: "human" or "llm" to select the dataset.
            k_sources: Number of sources to sample per run.
            times: Number of repeated runs.
            seed: Random seed for reproducibility.
            qid_col: Column for question IDs.
            source_col: Column for source identifiers.
            label_col: Column for labels/answers.
            label_space: Optional list of all possible labels.
            restrict_to_overlap_qids: If True, only include questions present
                in both human and LLM datasets.

        Returns:
            Tuple of (runs_df, summary_df, per_qid_df):
                - runs_df: Per-run entropy scores
                - summary_df: Mean, median, p10, p90 across runs
                - per_qid_df: Per-question entropy for each run
        """

        if which not in {"human", "llm"}:
            raise ValueError("which must be 'human' or 'llm'")

        base = self.human_df if which == "human" else self.llm_df

        if restrict_to_overlap_qids:
            base = base[base[qid_col].isin(self.overlap_qids)].copy()

        run_rows = []
        per_qid_runs = []

        for i in range(times):

            df_i = self._downsample_sources_once(
                base,
                k_sources=k_sources,
                source_col=source_col,
                seed=seed + i,
            )

            per_qid = self.disagreement_per_qid(
                df_i,
                qid_col=qid_col,
                source_col=source_col,
                label_col=label_col,
                label_space=label_space,
            )

            per_qid["run"] = i
            per_qid_runs.append(per_qid)

            overall = float(per_qid["entropy"].mean()) if len(per_qid) else np.nan

            run_rows.append({
                "run": i,
                "which": which,
                "k_sources": k_sources,
                "overall_entropy": overall,
            })

        runs_df = pd.DataFrame(run_rows)

        summary_df = pd.DataFrame([{
            "which": which,
            "k_sources": k_sources,
            "times": times,
            "mean": float(runs_df["overall_entropy"].mean()),
            "median": float(runs_df["overall_entropy"].median()),
            "p10": float(runs_df["overall_entropy"].quantile(0.10)),
            "p90": float(runs_df["overall_entropy"].quantile(0.90)),
        }])

        per_qid_df = pd.concat(per_qid_runs, ignore_index=True)

        return runs_df, summary_df, per_qid_df

    def compare_human_vs_llm(
        self,
        *,
        k_sources: int = 5,
        times: int = 100,
        seed: int = 42,
        **kwargs,
    ):
        """Compare disagreement between human and LLM labels.

        Runs the disagreement evaluation for both human and LLM data
        and returns results for comparison.

        Args:
            k_sources: Number of sources to sample per run.
            times: Number of repeated runs.
            seed: Random seed for reproducibility.
            **kwargs: Additional arguments passed to disagreement_many.

        Returns:
            Tuple of (human_results, llm_results), each being the tuple
            returned by disagreement_many.
        """
        h = self.disagreement_many(
            "human",
            k_sources=k_sources,
            times=times,
            seed=seed,
            **kwargs
        )

        l = self.disagreement_many(
            "llm",
            k_sources=k_sources,
            times=times,
            seed=seed,
            **kwargs
        )

        return h, l