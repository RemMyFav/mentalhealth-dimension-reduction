"""Internal disagreement evaluation using Jensen-Shannon Divergence."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


# -------------------------------------------------
# Internal Evaluator
# -------------------------------------------------

class InternalEvaluator:
    """Evaluate internal label consistency within sources.

    For each source (human or LLM), this evaluator measures how consistently
    the same source labels the same question across multiple repeats.
    Uses Jensen-Shannon Divergence (JSD) between the empirical distribution
    of labels and a one-hot distribution on the mode label.

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
        human_qids = set(self.human_df["qid"])
        llm_qids = set(self.llm_df["qid"])
        return sorted(human_qids & llm_qids)

    # ---------- JSD ----------
    @staticmethod
    def jsd(p: np.ndarray, q: np.ndarray) -> float:
        """Compute squared Jensen-Shannon Divergence between two distributions.

        Args:
            p: First probability distribution.
            q: Second probability distribution.

        Returns:
            Squared JSD value between 0 and 1.
        """
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        ps, qs = p.sum(), q.sum()
        if ps <= 0 or qs <= 0:
            return np.nan
        p = p / ps
        q = q / qs
        return float(jensenshannon(p, q, base=2.0) ** 2)

    # ---------- internal disagreement ----------
    def internal_jsd_per_qid(
        self,
        df: pd.DataFrame,
        *,
        qid_col: str = "qid",
        source_col: str = "source",
        label_col: str = "answer",
        label_space: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compute internal JSD per (source, question) pair.

        For each source and question combination, computes the empirical
        distribution of labels across repeats, then calculates JSD between
        that distribution and a one-hot on the mode label.

        Args:
            df: DataFrame with labels.
            qid_col: Column for question IDs.
            source_col: Column for source identifiers.
            label_col: Column for labels/answers.
            label_space: Optional list of all possible labels.

        Returns:
            pd.DataFrame with columns:
                - source: Source identifier
                - qid: Question identifier
                - n_repeats: Number of repeat answers
                - mode_label: Most common label
                - jsd_internal: JSD between empirical and mode distributions
        """
        x = df[[qid_col, source_col, label_col]].dropna().copy()
        x[label_col] = x[label_col].astype(str)

        if label_space is None:
            label_space = sorted(x[label_col].unique().tolist())

        label2idx = {lab: i for i, lab in enumerate(label_space)}

        rows = []
        for (src, qid), g in x.groupby([source_col, qid_col], sort=False):
            labels = g[label_col].tolist()
            counts = np.zeros(len(label_space), dtype=float)

            for lab in labels:
                idx = label2idx.get(lab)
                if idx is not None:
                    counts[idx] += 1

            if counts.sum() == 0:
                continue

            mode_idx = int(np.argmax(counts))
            q = np.zeros_like(counts)
            q[mode_idx] = 1.0

            rows.append({
                source_col: src,
                qid_col: qid,
                "n_repeats": int(len(labels)),
                "mode_label": label_space[mode_idx],
                "jsd_internal": self.jsd(counts, q),
            })

        return pd.DataFrame(rows)

    # ---------- NEW: source downsample ----------
    @staticmethod
    def _downsample_sources_once(
        df: pd.DataFrame,
        k_sources: int,
        *,
        source_col: str = "source",
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
        if len(sources) == 0:
            raise ValueError("No sources found.")
        k = min(k_sources, len(sources))
        picked = rng.choice(sources, size=k, replace=False)
        return df[df[source_col].isin(picked)].copy()

    # ---------- NEW: the exact workflow you described ----------
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
        """Run repeated internal JSD evaluation with source downsampling.

        For each run, samples k_sources and computes internal JSD per
        (source, question), then averages across questions. Repeats
        the process and returns summary statistics.

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
                - runs_df: Per-run overall JSD scores
                - summary_df: Mean, median, p10, p90 across runs
                - per_qid_df: Per-question mean JSD for each run
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
                base, k_sources=k_sources, source_col=source_col, seed=seed + i
            )

            # (source, qid) internal JSD
            jsd_src_qid = self.internal_jsd_per_qid(
                df_i,
                qid_col=qid_col,
                source_col=source_col,
                label_col=label_col,
                label_space=label_space,
            )

            # per-qid average across sampled sources
            jsd_qid = (
                jsd_src_qid
                .groupby(qid_col, as_index=False)["jsd_internal"]
                .mean()
                .rename(columns={"jsd_internal": "jsd_qid_mean"})
            )
            jsd_qid["run"] = i
            per_qid_runs.append(jsd_qid)

            # overall = average across qids
            overall = float(jsd_qid["jsd_qid_mean"].mean()) if len(jsd_qid) else np.nan
            run_rows.append({"run": i, "which": which, "k_sources": k_sources, "overall": overall})

        runs_df = pd.DataFrame(run_rows)

        summary_df = pd.DataFrame([{
            "which": which,
            "k_sources": k_sources,
            "times": times,
            "mean": float(runs_df["overall"].mean()),
            "median": float(runs_df["overall"].median()),
            "p10": float(runs_df["overall"].quantile(0.10)),
            "p90": float(runs_df["overall"].quantile(0.90)),
        }])

        per_qid_df = pd.concat(per_qid_runs, ignore_index=True)
        return runs_df, summary_df, per_qid_df

    # optional convenience
    def compare_human_vs_llm(
        self,
        *,
        k_sources: int = 5,
        times: int = 100,
        seed: int = 42,
        **kwargs,
    ):
        """Compare internal disagreement between human and LLM labels.

        Runs the internal JSD evaluation for both human and LLM data
        and returns results for comparison.

        Args:
            k_sources: Number of sources to sample per run.
            times: Number of repeated runs.
            seed: Random seed for reproducibility.
            **kwargs: Additional arguments passed to disagreement_many.

        Returns:
            Tuple of ((human_runs, human_sum, human_perqid), (llm_runs, llm_sum, llm_perqid)).
        """
        runs_h, sum_h, perqid_h = self.disagreement_many(
            "human", k_sources=k_sources, times=times, seed=seed, **kwargs
        )
        runs_l, sum_l, perqid_l = self.disagreement_many(
            "llm",   k_sources=k_sources, times=times, seed=seed, **kwargs
        )
        return (runs_h, sum_h, perqid_h), (runs_l, sum_l, perqid_l)