import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .loss import compute_total_loss


def compute_total_loss_lower_bound(
    df: pd.DataFrame,
    dimensions: list[str],
    qid_col: str = "qid",
    alpha: float = 1.0,
    beta: float = 1.0,
    margin: float = 0.1,
    lr: float = 0.1,
    steps: int = 300,
) -> dict:
    """
    Compute the empirical lower bound of the SAME total loss as the model:

        total_loss = alpha * BCEWithLogitsLoss + beta * pairwise_ranking_loss

    Rank convention:
        smaller positive number = higher rank
        0 = not selected

    For each qid:
        optimize a free logit vector s* to minimize the average total loss
        across all annotators of that qid.

    Returns dataset-level average total / bce / rank loss and metadata.
    """

    missing = [d for d in dimensions if d not in df.columns]
    if missing:
        raise ValueError(f"Missing target columns: {missing}")
    if qid_col not in df.columns:
        raise ValueError(f"Missing qid column: {qid_col}")

    bce_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    rank_loss_fn = nn.MarginRankingLoss(margin=margin, reduction="mean")

    total_total = 0.0
    total_bce = 0.0
    total_rank = 0.0

    total_items = 0
    total_pairs = 0
    n_qids = 0

    grouped = df.groupby(qid_col, sort=False)

    for _, g in grouped:
        y_all = g[dimensions].to_numpy(dtype=np.float32)
        n_ann, num_labels = y_all.shape

        if n_ann == 0:
            continue

        targets_t = torch.tensor(y_all, dtype=torch.float32)  # [n_ann, num_labels]

        # free logit vector for this qid
        s = torch.zeros((1, num_labels), dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([s], lr=lr)

        best_loss = None
        best_s = None

        for _ in range(steps):
            optimizer.zero_grad()

            # same free score vector for all annotators of this qid
            preds_t = s.repeat(n_ann, 1)

            loss, loss_bce, loss_rank, n_pairs = compute_total_loss(
                preds=preds_t,
                targets=targets_t,
                bce_loss_fn=bce_loss_fn,
                rank_loss_fn=rank_loss_fn,
                alpha=alpha,
                beta=beta,
            )

            loss.backward()
            optimizer.step()

            current = loss.item()
            if best_loss is None or current < best_loss:
                best_loss = current
                best_s = s.detach().clone()

        # evaluate again using best_s
        preds_t = best_s.repeat(n_ann, 1)
        loss, loss_bce, loss_rank, n_pairs = compute_total_loss(
            preds=preds_t,
            targets=targets_t,
            bce_loss_fn=bce_loss_fn,
            rank_loss_fn=rank_loss_fn,
            alpha=alpha,
            beta=beta,
        )

        # BCE and total are averaged per item (annotator-row)
        total_total += loss.item() * n_ann
        total_bce += loss_bce.item() * n_ann
        total_items += n_ann

        # rank is averaged per pair
        if n_pairs > 0:
            total_rank += loss_rank.item() * n_pairs
            total_pairs += n_pairs

        n_qids += 1

    avg_total = total_total / total_items if total_items > 0 else 0.0
    avg_bce = total_bce / total_items if total_items > 0 else 0.0
    avg_rank = total_rank / total_pairs if total_pairs > 0 else 0.0

    return {
        "total_lower_bound": avg_total,
        "bce_lower_bound": avg_bce,
        "rank_lower_bound": avg_rank,
        "n_items": total_items,
        "n_pairs": total_pairs,
        "n_qids": n_qids,
        "alpha": alpha,
        "beta": beta,
        "margin": margin,
        "steps": steps,
        "lr": lr,
    }