import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from itertools import product

class RankDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class LinearRankClassifier(nn.Module):
    def __init__(self, input_dim: int, num_labels: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        return self.linear(x)

class HumanRankClassifier:
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

    def _build_targets(self, df: pd.DataFrame) -> np.ndarray:
        missing = [d for d in self.dimensions if d not in df.columns]
        if missing:
            raise ValueError(f"Missing target columns: {missing}")

        return df[self.dimensions].to_numpy(dtype=np.float32)

    def _build_selection_targets(self, Y: torch.Tensor) -> torch.Tensor:
        return (Y > 0).float()

    def evaluate_human_baseline_loss(
    self,
    df: pd.DataFrame,
    qid_col: str = "qid",
    ) -> dict:
        """
        Compute leave-one-out human baseline using the same loss definition as the model:
        - BCE on selection targets (Y > 0)
        - ranking only among positive labels
        - no ranking against 0
        - same alpha / beta / margin
        """

        missing = [d for d in self.dimensions if d not in df.columns]
        if missing:
            raise ValueError(f"Missing target columns: {missing}")
        if qid_col not in df.columns:
            raise ValueError(f"Missing qid column: {qid_col}")

        total_bce = 0.0
        total_rank = 0.0
        total_items = 0
        total_pairs = 0

        grouped = df.groupby(qid_col, sort=False)

        for _, g in grouped:
            Y = g[self.dimensions].to_numpy(dtype=np.float32)
            n_ann, _ = Y.shape

            if n_ann < 2:
                continue

            for i in range(n_ann):
                target = Y[i]
                others = np.delete(Y, i, axis=0)

                # ---- BCE part ----
                target_sel = (target > 0).astype(np.float32)
                pred_sel_prob = (others > 0).astype(np.float32).mean(axis=0)

                eps = 1e-6
                pred_sel_prob = np.clip(pred_sel_prob, eps, 1.0 - eps)
                pred_sel_logits = np.log(pred_sel_prob / (1.0 - pred_sel_prob))

                target_sel_t = torch.tensor(target_sel, dtype=torch.float32)
                pred_sel_logits_t = torch.tensor(pred_sel_logits, dtype=torch.float32)

                bce = float(
                    self.bce_loss(
                        pred_sel_logits_t,
                        target_sel_t,
                    ).item()
                )

                total_bce += bce
                total_items += 1

                # ---- Rank part ----
                pred_scores = others.mean(axis=0)

                pred_scores_t = torch.tensor(pred_scores, dtype=torch.float32).unsqueeze(0)
                target_t = torch.tensor(target, dtype=torch.float32).unsqueeze(0)

                rank_loss_value, n_pairs = self._ranking__pairwiseloss(
                    preds=pred_scores_t,
                    targets=target_t,
                )

                if n_pairs > 0:
                    total_rank += rank_loss_value.item() * n_pairs
                    total_pairs += n_pairs

        avg_bce = total_bce / total_items if total_items > 0 else 0.0
        avg_rank = total_rank / total_pairs if total_pairs > 0 else 0.0
        avg_total = self.alpha * avg_bce + self.beta * avg_rank

        return {
            "loss": avg_total,
            "bce": avg_bce,
            "rank": avg_rank,
            "n_items": total_items,
            "n_pairs": total_pairs,
        }

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

    def _compute_total_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        selection_targets = self._build_selection_targets(targets)

        loss_bce = self.bce_loss(preds, selection_targets)
        loss_rank, n_pairs = self._pairwise_ranking_loss(preds, targets)

        total_loss = self.alpha * loss_bce + self.beta * loss_rank
        return total_loss, loss_bce, loss_rank, n_pairs

    def evaluate_loss(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        batch_size: int = 64,
    ) -> dict:
        if self.model is None:
            raise ValueError("Model is not trained yet. Call fit(...) first.")

        texts = df[text_col].astype(str).tolist()
        X = self._encode_texts(texts, show_progress_bar=False)
        Y = self._build_targets(df)

        dataset = RankDataset(X, Y)
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
                loss, loss_bce, loss_rank, n_pairs = self._compute_total_loss(preds, yb)

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

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        text_col: str = "text",
        epochs: int = 20,
        batch_size: int = 16,
        lr: float = 1e-3,
    ):
        self.train_losses = []
        self.val_losses = []

        texts = train_df[text_col].astype(str).tolist()
        X = self._encode_texts(texts, show_progress_bar=True)
        Y = self._build_targets(train_df)

        dataset = RankDataset(X, Y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = LinearRankClassifier(
            input_dim=X.shape[1],
            num_labels=Y.shape[1],
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

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

                loss, loss_bce, loss_rank, n_pairs = self._compute_total_loss(preds, yb)

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
                val_metrics = self.evaluate_loss(val_df, text_col=text_col, batch_size=batch_size)
                self.val_losses.append(val_metrics)

                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"(bce={train_metrics['bce']:.4f}, rank={train_metrics['rank']:.4f}, pairs={train_metrics['n_pairs']}) | "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"(bce={val_metrics['bce']:.4f}, rank={val_metrics['rank']:.4f}, pairs={val_metrics['n_pairs']})"
                )
            else:
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"(bce={train_metrics['bce']:.4f}, rank={train_metrics['rank']:.4f}, pairs={train_metrics['n_pairs']})"
                )

    def predict_scores(self, texts: list[str]) -> np.ndarray:
        X = self._encode_texts(texts, show_progress_bar=False)
        X = torch.tensor(X, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            scores = self.model(X).cpu().numpy()

        return scores

    def predict_rankings(self, texts: list[str]) -> list[list[str]]:
        scores = self.predict_scores(texts)

        out = []
        for row in scores:
            idx = np.argsort(row)[::-1]
            out.append([self.idx2dim[i] for i in idx])

        return out

    def predict_selection(self, texts: list[str], threshold: float = 0.5) -> list[list[str]]:
        scores = self.predict_scores(texts)
        probs = 1 / (1 + np.exp(-scores))  # sigmoid

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

    def evaluate_human_baseline_loss(
        self,
        df: pd.DataFrame,
        qid_col: str = "qid",
    ) -> dict:
        """
        Compute leave-one-out human baseline using the same loss definition as the model:
        - BCE on selection targets (Y > 0)
        - ranking only among positive labels
        - no ranking against 0
        - same alpha / beta / margin

        Assumes:
        - each row = one annotator for one qid
        - multiple rows per qid
        """

        missing = [d for d in self.dimensions if d not in df.columns]
        if missing:
            raise ValueError(f"Missing target columns: {missing}")
        if qid_col not in df.columns:
            raise ValueError(f"Missing qid column: {qid_col}")

        total_bce = 0.0
        total_rank = 0.0
        total_items = 0
        total_pairs = 0

        grouped = df.groupby(qid_col, sort=False)

        for _, g in grouped:
            Y = g[self.dimensions].to_numpy(dtype=np.float32)   # [n_annotators, n_dims]
            n_ann, n_dims = Y.shape

            # need at least 2 annotators for leave-one-out
            if n_ann < 2:
                continue

            for i in range(n_ann):
                target = Y[i]                      # one annotator target
                others = np.delete(Y, i, axis=0)  # leave-one-out others

                # -------------------------
                # BCE part: same logic as model
                # selection target = (Y > 0)
                # -------------------------
                target_sel = (target > 0).astype(np.float32)
                pred_sel_prob = (others > 0).astype(np.float32).mean(axis=0)

                eps = 1e-6
                pred_sel_prob = np.clip(pred_sel_prob, eps, 1.0 - eps)

                pred_sel_logits = np.log(pred_sel_prob / (1.0 - pred_sel_prob))

                target_sel_t = torch.tensor(target_sel, dtype=torch.float32)
                pred_sel_logits_t = torch.tensor(pred_sel_logits, dtype=torch.float32)

                bce = float(
                    self.bce_loss(
                        pred_sel_logits_t,
                        target_sel_t,
                    ).item()
                )

                total_bce += bce
                total_items += 1

                # -------------------------
                # Ranking part: same logic as model
                # - only positive labels
                # - no ranking against 0
                # - pairwise margin loss
                # -------------------------
                pred_scores = others.mean(axis=0)   # consensus prediction scores
                pos_idx = np.where(target > 0)[0]

                if len(pos_idx) >= 2:
                    for a in range(len(pos_idx)):
                        for c in range(a + 1, len(pos_idx)):
                            p = pos_idx[a]
                            q = pos_idx[c]

                            if target[p] == target[q]:
                                continue

                            if target[p] > target[q]:
                                hi, lo = p, q
                            else:
                                hi, lo = q, p

                            diff = pred_scores[hi] - pred_scores[lo]
                            pair_loss = max(0.0, self.rank_loss.margin - diff)

                            total_rank += pair_loss
                            total_pairs += 1

        avg_bce = total_bce / total_items if total_items > 0 else 0.0
        avg_rank = total_rank / total_pairs if total_pairs > 0 else 0.0
        avg_total = self.alpha * avg_bce + self.beta * avg_rank

        return {
            "loss": avg_total,
            "bce": avg_bce,
            "rank": avg_rank,
            "n_items": total_items,
            "n_pairs": total_pairs,
        }

    def scores_to_ranked_vector(
        self,
        scores: np.ndarray,
        dimensions: list[str] | None = None,
        threshold: float = 0.5,
    ) -> dict:
        """
        Convert model scores into rank-style labels with zeros.

        Output format:
            1 = best
            2 = second
            ...
            0 = not selected
        """

        if dimensions is None:
            dimensions = self.dimensions

        probs = 1 / (1 + np.exp(-scores))

        # sort dimensions by score descending
        order = np.argsort(scores)[::-1]

        out = {dim: 0 for dim in dimensions}

        rank = 1

        for i in order:
            if probs[i] >= threshold:
                out[dimensions[i]] = rank
                rank += 1

        return out


    def evaluate_postprocess_l1(
        self,
        df: pd.DataFrame,
        dimensions: list[str] | None = None,
        text_col: str = "text",
        threshold: float = 0.5,
    ) -> float:
        """
        Evaluate final postprocessed prediction against ground truth using mean L1 distance.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Call fit(...) first.")

        if dimensions is None:
            dimensions = self.dimensions

        texts = df[text_col].astype(str).tolist()
        scores_all = self.predict_scores(texts)

        total_l1 = 0.0

        for scores, (_, row) in zip(scores_all, df.iterrows()):
            pred_dict = self.scores_to_ranked_vector(
                scores=scores,
                dimensions=dimensions,
                threshold=threshold,
            )

            y_pred = np.array([pred_dict[d] for d in dimensions], dtype=np.float32)
            y_true = row[dimensions].to_numpy(dtype=np.float32)

            total_l1 += np.abs(y_pred - y_true).sum()

        return total_l1 / len(df)


    def search_best_threshold(
    self,
    val_df: pd.DataFrame,
    dimensions: list[str] | None = None,
    text_col: str = "text",
    thresholds: list[float] | None = None,
    ) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model is not trained yet. Call fit(...) first.")

        if dimensions is None:
            dimensions = self.dimensions

        if thresholds is None:
            thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

        texts = val_df[text_col].astype(str).tolist()
        scores_all = self.predict_scores(texts)
        y_true_all = val_df[dimensions].to_numpy(dtype=np.float32)

        rows = []

        for th in thresholds:
            total_l1 = 0.0

            for scores, y_true in zip(scores_all, y_true_all):
                pred_dict = self.scores_to_ranked_vector(
                    scores=scores,
                    dimensions=dimensions,
                    threshold=th,
                )
                y_pred = np.array([pred_dict[d] for d in dimensions], dtype=np.float32)
                total_l1 += np.abs(y_pred - y_true).sum()

            val_l1 = total_l1 / len(val_df)

            rows.append({
                "threshold": th,
                "val_l1": val_l1,
            })

        return pd.DataFrame(rows).sort_values("val_l1").reset_index(drop=True)
    
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

        result_df = (
            pd.DataFrame(rows)
            .sort_values("val_loss")
            .reset_index(drop=True)
        )

        return result_df
    
    def _pairwise_ranking_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """
        Ranking only among positive labels:
        e.g. 3 > 2, 3 > 1, 2 > 1
        No comparison against 0.

        Returns
        -------
        loss: torch.Tensor
        n_pairs: int
        """
        losses = []

        batch_size, num_classes = targets.shape

        for b in range(batch_size):
            y = targets[b]
            s = preds[b]

            pos_idx = torch.where(y > 0)[0]

            if len(pos_idx) < 2:
                continue

            for a in range(len(pos_idx)):
                for c in range(a + 1, len(pos_idx)):
                    i = pos_idx[a].item()
                    j = pos_idx[c].item()

                    if y[i] == y[j]:
                        continue

                    if y[i] > y[j]:
                        hi, lo = i, j
                    else:
                        hi, lo = j, i

                    target = torch.tensor([1.0], device=preds.device)

                    losses.append(
                        self.rank_loss(
                            s[hi].unsqueeze(0),
                            s[lo].unsqueeze(0),
                            target,
                        )
                    )

        if len(losses) == 0:
            return torch.tensor(0.0, device=preds.device), 0

        return torch.stack(losses).mean(), len(losses)
    
    def scores_to_ranked_vector_threshold_gap(
        self,
        scores: np.ndarray,
        dimensions: list[str] | None = None,
        threshold: float = 0.45,
        gap_threshold: float = 0.1,
    ) -> dict:
        """
        Hybrid postprocess:
        1. always keep top-1
        2. among remaining labels, keep those with prob >= threshold
        3. within kept labels, if adjacent score gap > gap_threshold, cut there

        Output:
            1 = best
            2 = second
            ...
            0 = not selected
        """
        if dimensions is None:
            dimensions = self.dimensions

        probs = 1 / (1 + np.exp(-scores))
        order = np.argsort(scores)[::-1]

        out = {dim: 0 for dim in dimensions}

        if len(order) == 0:
            return out

        # always keep top-1
        candidate_order = [order[0]]

        # then apply threshold to the rest
        for i in order[1:]:
            if probs[i] >= threshold:
                candidate_order.append(i)

        keep_k = len(candidate_order)

        # optional gap-based early cut
        if len(candidate_order) >= 2:
            candidate_scores = np.array([scores[i] for i in candidate_order])

            for i in range(len(candidate_scores) - 1):
                gap = candidate_scores[i] - candidate_scores[i + 1]
                if gap > gap_threshold:
                    keep_k = i + 1
                    break

        for rank, idx in enumerate(candidate_order[:keep_k], start=1):
            out[dimensions[idx]] = rank

        return out
    
    def evaluate_postprocess_l1(
    self,
    df: pd.DataFrame,
    dimensions: list[str] | None = None,
    text_col: str = "text",
    method: str = "threshold",
    threshold: float = 0.5,
    gap_threshold: float = 0.1,
    ) -> float:
        """
        Evaluate final postprocessed prediction against ground truth using mean L1 distance.

        method:
            - "threshold"
            - "threshold_gap"
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Call fit(...) first.")

        if dimensions is None:
            dimensions = self.dimensions

        texts = df[text_col].astype(str).tolist()
        scores_all = self.predict_scores(texts)

        total_l1 = 0.0

        for scores, (_, row) in zip(scores_all, df.iterrows()):
            if method == "threshold":
                pred_dict = self.scores_to_ranked_vector(
                    scores=scores,
                    dimensions=dimensions,
                    threshold=threshold,
                )
            elif method == "threshold_gap":
                pred_dict = self.scores_to_ranked_vector_threshold_gap(
                    scores=scores,
                    dimensions=dimensions,
                    threshold=threshold,
                    gap_threshold=gap_threshold,
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            y_pred = np.array([pred_dict[d] for d in dimensions], dtype=np.float32)
            y_true = row[dimensions].to_numpy(dtype=np.float32)

            total_l1 += np.abs(y_pred - y_true).sum()

        return total_l1 / len(df)
    
    def search_best_postprocess(
        self,
        val_df: pd.DataFrame,
        dimensions: list[str] | None = None,
        text_col: str = "text",
        thresholds: list[float] | None = None,
        gap_thresholds: list[float] | None = None,
    ) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model is not trained yet. Call fit(...) first.")

        if dimensions is None:
            dimensions = self.dimensions

        if thresholds is None:
            thresholds = [0.35, 0.4, 0.45, 0.5, 0.55]

        if gap_thresholds is None:
            gap_thresholds = [0.03, 0.05, 0.08, 0.1, 0.15]

        rows = []

        # baseline: threshold only
        for th in thresholds:
            val_l1 = self.evaluate_postprocess_l1(
                df=val_df,
                dimensions=dimensions,
                text_col=text_col,
                method="threshold",
                threshold=th,
            )
            rows.append({
                "method": "threshold",
                "threshold": th,
                "gap_threshold": None,
                "val_l1": val_l1,
            })

        # hybrid: threshold + gap
        for th, gap_th in product(thresholds, gap_thresholds):
            val_l1 = self.evaluate_postprocess_l1(
                df=val_df,
                dimensions=dimensions,
                text_col=text_col,
                method="threshold_gap",
                threshold=th,
                gap_threshold=gap_th,
            )
            rows.append({
                "method": "threshold_gap",
                "threshold": th,
                "gap_threshold": gap_th,
                "val_l1": val_l1,
            })

        return pd.DataFrame(rows).sort_values("val_l1").reset_index(drop=True)