import copy
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from itertools import product

from .model import LinearRankClassifier
from .dataset import RankDataset
from .loss import compute_soft_selection_loss


class SelectionClassifier:
    """
    Selection-only classifier with soft labels.

    Learns:
        probability that each dimension should be selected

    Target convention:
        each target is already a soft value in [0, 1]
    """

    def __init__(
        self,
        dimensions: list[str],
        embedding_model: str = "all-MiniLM-L6-v2",
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        self.dimensions = dimensions
        self.dim2idx = {d: i for i, d in enumerate(dimensions)}
        self.idx2dim = {i: d for d, i in self.dim2idx.items()}

        self.embedding_model = embedding_model
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.embedder = SentenceTransformer(embedding_model)
        self.model = None

        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

        self.train_losses = []
        self.val_losses = []

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
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(epochs):
            self.model.train()

            total_loss = 0.0
            total_samples = 0

            for xb, yb in loader:
                optimizer.zero_grad()

                preds = self.model(xb)

                loss, loss_bce = compute_soft_selection_loss(
                    preds=preds,
                    targets=yb,
                    bce_loss_fn=self.bce_loss,
                )

                loss.backward()
                optimizer.step()

                bs = xb.size(0)
                total_loss += loss.item() * bs
                total_samples += bs

            train_metrics = {
                "loss": total_loss / total_samples if total_samples > 0 else 0.0,
                "bce": total_loss / total_samples if total_samples > 0 else 0.0,
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
                    f"(bce={train_metrics['bce']:.4f}) | "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"(bce={val_metrics['bce']:.4f})"
                )

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
                    f"(bce={train_metrics['bce']:.4f})"
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
        total_samples = 0

        with torch.no_grad():
            for xb, yb in loader:
                preds = self.model(xb)

                loss, loss_bce = compute_soft_selection_loss(
                    preds=preds,
                    targets=yb,
                    bce_loss_fn=self.bce_loss,
                )

                bs = xb.size(0)
                total_loss += loss.item() * bs
                total_samples += bs

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        return {
            "loss": avg_loss,
            "bce": avg_loss,
        }

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

    def predict_selection(
        self,
        texts: list[str],
        threshold: float = 0.5,
    ) -> list[list[str]]:
        probs = self.predict_probs(texts)

        out = []
        for row in probs:
            selected = [self.idx2dim[i] for i, p in enumerate(row) if p >= threshold]
            out.append(selected)

        return out

    def predict_topk(
        self,
        texts: list[str],
        k: int = 8,
    ) -> list[list[str]]:
        scores = self.predict_scores(texts)

        out = []
        for row in scores:
            idx = np.argsort(row)[::-1][:k]
            out.append([self.idx2dim[i] for i in idx])

        return out


    def search_train_hyperparams(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        *,
        lr_list=(1e-3, 3e-4, 1e-4),
        hidden_dim_list=(512,),
        dropout_list=(0.1,),
        epochs: int = 20,
        batch_size: int = 16,
        text_col: str = "text",
        patience: int | None = 3,
        save_csv: str | None = None,
    ) -> pd.DataFrame:

        rows = []

        existing_df = None
        if save_csv is not None and os.path.exists(save_csv):
            existing_df = pd.read_csv(save_csv)
            print(f"Loaded existing search results: {len(existing_df)} rows")

        for lr, hidden_dim, dropout in product(lr_list, hidden_dim_list, dropout_list):

            if existing_df is not None:
                mask = (
                    (existing_df["lr"] == lr) &
                    (existing_df["hidden_dim"] == hidden_dim) &
                    (existing_df["dropout"] == dropout)
                )
                if mask.any():
                    print(f"Skipping existing config: lr={lr}, hidden_dim={hidden_dim}, dropout={dropout}")
                    continue

            print(f"\n=== lr={lr}, hidden_dim={hidden_dim}, dropout={dropout} ===")

            clf = self.__class__(
                dimensions=self.dimensions,
                embedding_model=self.embedding_model,
                hidden_dim=hidden_dim,
                dropout=dropout,
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

            row = {
                "lr": lr,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "val_loss": val_metrics["loss"],
                "val_bce": val_metrics["bce"],
            }

            rows.append(row)

            print(
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_bce={val_metrics['bce']:.4f}"
            )

            if save_csv is not None:
                df_now = pd.DataFrame(rows)

                if existing_df is not None:
                    df_now = pd.concat([existing_df, df_now], ignore_index=True)

                df_now = df_now.drop_duplicates(
                    subset=["lr", "hidden_dim", "dropout"],
                    keep="first",
                )

                df_now.to_csv(save_csv, index=False)


        if save_csv is not None and os.path.exists(save_csv):
            df_final = pd.read_csv(save_csv)
        else:
            df_final = pd.DataFrame(rows)

        df_final = df_final.sort_values("val_loss").reset_index(drop=True)

        if save_csv is not None:
            df_final.to_csv(save_csv, index=False)

        return df_final