import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from typing import Union

class SemanticCluster:
    def __init__(self, embedder, k=8, random_state=42):
        """
        embedder: any model with .encode(texts, convert_to_numpy=True)
        """
        self.embedder = embedder
        self.k = k
        self.random_state = random_state

        self.embeddings = None
        self.df = None
        self.cluster_id = None
        self.centers = None
        self.kmeans = None

    # --------------------------------------------------
    # 1) Fit everything
    # --------------------------------------------------
    def fit(self, df: pd.DataFrame, text_col="text"):
        texts = df[text_col].astype(str).tolist()

        # 🔹 embedding
        self.embeddings = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True  # 强烈推荐
        )

        # 🔹 clustering
        self.kmeans = KMeans(
            n_clusters=self.k,
            random_state=self.random_state,
            n_init="auto"
        )
        self.cluster_id = self.kmeans.fit_predict(self.embeddings)
        self.centers = self.kmeans.cluster_centers_

        self.df = df.copy()
        self.df["cluster_id"] = self.cluster_id

        return self.df

    # --------------------------------------------------
    # 2) Cluster representatives
    # --------------------------------------------------
    def get_representatives(self, top_n=10):
        rows = []

        for c in range(self.k):
            idx = np.where(self.cluster_id == c)[0]
            if len(idx) == 0:
                continue

            sims = cosine_similarity(
                self.embeddings[idx],
                self.centers[c].reshape(1, -1)
            ).reshape(-1)

            order = np.argsort(sims)[::-1][:top_n]
            top_idx = idx[order]

            sub = self.df.iloc[top_idx][["qid", "dataset", "text", "cluster_id"]].copy()
            sub["sim_to_center"] = sims[order]
            rows.append(sub)

        return pd.concat(rows, ignore_index=True)

    # --------------------------------------------------
    # 3) Query similar sentences
    # --------------------------------------------------
    def query_similar(self, i, top_k=6):
        sims = cosine_similarity(
            self.embeddings[i].reshape(1, -1),
            self.embeddings
        )[0]

        order = sims.argsort()[::-1][:top_k]
        out = self.df.iloc[order][["qid", "dataset", "text", "cluster_id"]].copy()
        out["sim"] = sims[order]
        return out
    
    from pathlib import Path

    def save_cluster(self, out_path: Union[str, Path] = None) -> Path:
        if self.df is None:
            raise ValueError("DataFrame is not set.")

        df_out = self.df.sort_values(
            by=["cluster_id", "dataset", "qid"],
            ascending=[True, True, True]
        )

        # ✅ 统一转成 Path
        out_path = Path(out_path) if out_path is not None else Path("./temp_result/cluster.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        df_out.to_csv(out_path, index=False)
        return out_path
    
    def map_cluster_type(
    self,
    cluster_types: list[str],
    new_col: str = "cluster_type"
    ):
        """
        cluster_types: list where index = cluster_id, value = type name
        """
        if self.df is None:
            raise ValueError("DataFrame is not set. Run fit() first.")

        if len(cluster_types) < self.k:
            raise ValueError(
                f"cluster_types length ({len(cluster_types)}) < k ({self.k})"
            )

        self.df[new_col] = self.df["cluster_id"].map(
            lambda x: cluster_types[x]
        )

        return self.df
    
