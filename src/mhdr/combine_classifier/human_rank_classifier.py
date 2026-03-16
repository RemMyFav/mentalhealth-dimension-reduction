import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from itertools import product

from .model import LinearRankClassifier
from .dataset import RankDataset
from .loss import compute_total_loss


class HumanRankClassifier:
    """
    Rank convention used everywhere in this class:
    - smaller positive number = higher rank
    - 0 = not selected

    Example:
        1 > 2 > 3 > 0
    """

    def __init__(
        self,
        dimensions: list[str],
        embedding_model: str = "all-MiniLM-L6-v2",
        alpha: float = 1.0,
        beta: float = 1.0,
        margin: float = 0.1,
    ):
        self.dimensions = dimensions
        self.dim2idx = {d: i for i, d in enumerate(dimensions)}
        self.idx2dim = {i: d for d, i in self.dim2idx.items()}

        self.embedding_model = embedding_model
        self.embedder = SentenceTransformer(embedding_model)
        self.model = None

        self.alpha = alpha
        self.beta = beta

        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.rank_loss = nn.MarginRankingLoss(margin=margin, reduction="mean")

        self.train_losses = []
        self.val_losses = []

    # =========================
    # target / encoding helpers
    # =========================
    def _build_targets(self, df: pd.DataFrame) -> np.ndarray:
        missing = [d for d in self.dimensions if d not in df.columns]
        if missing:
            raise ValueError(f"Missing target columns: {missing}")
        return df[self.dimensions].to_numpy(dtype=np.float32)

    def _encode_texts(
        self,
        texts: list[str],
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        return self.embedder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )

    # =========================
    # training / evaluation
    # =========================
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        text_col: str = "text",
        epochs: int = 20,
        batch_size: int = 16,
        lr: float = 1e-3,
        patience: int | None = 3,
    ):
        self.train_losses = []
        self.val_losses = []

        texts = train_df[text_col].astype(str).tolist()
        x = self._encode_texts(texts, show_progress_bar=True)
        y = self._build_targets(train_df)

        dataset = RankDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = LinearRankClassifier(
            input_dim=x.shape[1],
            num_labels=y.shape[1],
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(epochs):
            self.model.train()

            total_loss = 0.0
            total_bce = 0.0
            total_rank = 0.0
            total_samples = 0
            total_pairs = 0

            for xb, yb in loader:
                optimizer.zero_grad()

                preds = self.model(xb)
                loss, loss_bce, loss_rank, n_pairs = compute_total_loss(
                    preds=preds,
                    targets=yb,
                    bce_loss_fn=self.bce_loss,
                    rank_loss_fn=self.rank_loss,
                    alpha=self.alpha,
                    beta=self.beta,
                )

                loss.backward()
                optimizer.step()

                bs = xb.size(0)
                total_loss += loss.item() * bs
                total_bce += loss_bce.item() * bs
                total_samples += bs

                if n_pairs > 0:
                    total_rank += loss_rank.item() * n_pairs
                    total_pairs += n_pairs

            train_metrics = {
                "loss": total_loss / total_samples if total_samples > 0 else 0.0,
                "bce": total_bce / total_samples if total_samples > 0 else 0.0,
                "rank": total_rank / total_pairs if total_pairs > 0 else 0.0,
                "n_pairs": total_pairs,
            }
            self.train_losses.append(train_metrics)

            if val_df is not None:
                val_metrics = self.evaluate_loss(
                    val_df,
                    text_col=text_col,
                    batch_size=batch_size,
                )
                self.val_losses.append(val_metrics)

                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"(bce={train_metrics['bce']:.4f}, rank={train_metrics['rank']:.4f}, pairs={train_metrics['n_pairs']}) | "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"(bce={val_metrics['bce']:.4f}, rank={val_metrics['rank']:.4f}, pairs={val_metrics['n_pairs']})"
                )

                # ===== EARLY STOPPING =====
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    best_state = copy.deepcopy(self.model.state_dict())
                    wait = 0
                else:
                    wait += 1

                if patience is not None and wait >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break

            else:
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"(bce={train_metrics['bce']:.4f}, rank={train_metrics['rank']:.4f}, pairs={train_metrics['n_pairs']})"
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def evaluate_loss(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        batch_size: int = 64,
    ) -> dict:
        if self.model is None:
            raise ValueError("Model is not trained yet. Call fit(...) first.")

        texts = df[text_col].astype(str).tolist()
        x = self._encode_texts(texts, show_progress_bar=False)
        y = self._build_targets(df)

        dataset = RankDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()

        total_loss = 0.0
        total_bce = 0.0
        total_rank = 0.0
        total_samples = 0
        total_pairs = 0

        with torch.no_grad():
            for xb, yb in loader:
                preds = self.model(xb)
                loss, loss_bce, loss_rank, n_pairs = compute_total_loss(
                    preds=preds,
                    targets=yb,
                    bce_loss_fn=self.bce_loss,
                    rank_loss_fn=self.rank_loss,
                    alpha=self.alpha,
                    beta=self.beta,
                )

                bs = xb.size(0)
                total_loss += loss.item() * bs
                total_bce += loss_bce.item() * bs
                total_samples += bs

                if n_pairs > 0:
                    total_rank += loss_rank.item() * n_pairs
                    total_pairs += n_pairs

        return {
            "loss": total_loss / total_samples if total_samples > 0 else 0.0,
            "bce": total_bce / total_samples if total_samples > 0 else 0.0,
            "rank": total_rank / total_pairs if total_pairs > 0 else 0.0,
            "n_pairs": total_pairs,
        }

    # =========================
    # prediction
    # =========================
    def predict_scores(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained yet. Call fit(...) first.")

        x = self._encode_texts(texts, show_progress_bar=False)
        x = torch.tensor(x, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            scores = self.model(x).cpu().numpy()

        return scores

    def predict_probs(self, texts: list[str]) -> np.ndarray:
        scores = self.predict_scores(texts)
        return 1 / (1 + np.exp(-scores))

    def predict_rankings(self, texts: list[str]) -> list[list[str]]:
        scores = self.predict_scores(texts)

        out = []
        for row in scores:
            idx = np.argsort(row)[::-1]
            out.append([self.idx2dim[i] for i in idx])

        return out

    def predict_selection(self, texts: list[str], threshold: float = 0.5) -> list[list[str]]:
        probs = self.predict_probs(texts)

        out = []
        for row in probs:
            selected = [self.idx2dim[i] for i, p in enumerate(row) if p >= threshold]
            out.append(selected)

        return out

    def predict_distribution(self, texts: list[str]) -> np.ndarray:
        scores = self.predict_scores(texts)

        scores = np.maximum(scores, 0)
        row_sums = scores.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return scores / row_sums

    # =========================
    # hyperparam search
    # =========================
    def search_train_hyperparams(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        *,
        alpha_list=(1.0,),
        beta_list=(0.5, 1.0, 1.5, 2.0, 3.0, 5.0),
        lr_list=(1e-3, 3e-4, 1e-4),
        margin_list=(0.1, 0.3, 0.5, 0.7),
        epochs: int = 20,
        batch_size: int = 16,
        text_col: str = "text",
        patience: int | None = 3,
    ) -> pd.DataFrame:
        rows = []

        for alpha, beta, lr, margin in product(alpha_list, beta_list, lr_list, margin_list):
            print(f"\n=== alpha={alpha}, beta={beta}, lr={lr}, margin={margin} ===")

            clf = self.__class__(
                dimensions=self.dimensions,
                embedding_model=self.embedding_model,
                alpha=alpha,
                beta=beta,
                margin=margin,
            )

            clf.fit(
                train_df=train_df,
                val_df=val_df,
                text_col=text_col,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                patience=patience,
            )

            val_metrics = clf.evaluate_loss(
                val_df,
                text_col=text_col,
                batch_size=64,
            )

            rows.append({
                "alpha": alpha,
                "beta": beta,
                "lr": lr,
                "margin": margin,
                "val_loss": val_metrics["loss"],
                "val_bce": val_metrics["bce"],
                "val_rank": val_metrics["rank"],
            })

            print(
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_bce={val_metrics['bce']:.4f} | "
                f"val_rank={val_metrics['rank']:.4f}"
            )

        return pd.DataFrame(rows).sort_values("val_loss").reset_index(drop=True)