import numpy as np
import pandas as pd


class PopulationalEvaluator:
    """
    Population-wise disagreement (no repeats):
      - for each qid: look at labels across sources (humans or LLM models)
      - compute normalized entropy per qid
      - downsample sources to k and repeat many times
    """

    def __init__(self, human_df: pd.DataFrame, llm_df: pd.DataFrame):
        self.human_df = human_df.copy()
        self.llm_df = llm_df.copy()

        self.overlap_qids = self._get_overlap_qids()
        if len(self.overlap_qids) == 0:
            raise ValueError("No overlapping qids found between human and LLM data.")

    # ---------- overlap ----------
    def _get_overlap_qids(self) -> list[str]:
        return sorted(set(self.human_df["qid"]) & set(self.llm_df["qid"]))

    # ---------- entropy ----------
    @staticmethod
    def _entropy_norm(counts: np.ndarray) -> float:
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
        """
        Population-wise entropy:
          - each run: sample k_sources sources
          - compute per-qid entropy across sampled sources
          - overall per run = mean(entropy over qids)
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