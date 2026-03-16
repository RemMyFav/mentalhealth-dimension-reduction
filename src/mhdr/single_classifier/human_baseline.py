import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .loss import compute_soft_selection_loss, compute_ranking_loss


def compute_bce_lower_bound(
    df: pd.DataFrame,
    dimensions: list[str],
    qid_col: str = "qid",
    lr: float = 0.1,
    steps: int = 300,
):
    """
    Empirical lower bound for soft-label BCE.

    Assumes:
        - each row is already one qid
        - each dimension target is a soft value in [0, 1]
    """

    bce_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    total_bce = 0.0
    total_items = 0
    n_qids = 0

    grouped = df.groupby(qid_col, sort=False)

    for _, g in grouped:
        y_all = g[dimensions].to_numpy(dtype=np.float32)
        n_rows, num_labels = y_all.shape

        if n_rows == 0:
            continue

        targets_t = torch.tensor(y_all, dtype=torch.float32)

        s = torch.zeros(
            (1, num_labels),
            dtype=torch.float32,
            requires_grad=True,
        )

        optimizer = torch.optim.Adam([s], lr=lr)

        best_loss = None
        best_s = None

        for _ in range(steps):
            optimizer.zero_grad()

            preds = s.repeat(n_rows, 1)

            loss, loss_bce = compute_soft_selection_loss(
                preds=preds,
                targets=targets_t,
                bce_loss_fn=bce_loss_fn,
            )

            loss.backward()
            optimizer.step()

            if best_loss is None or loss.item() < best_loss:
                best_loss = loss.item()
                best_s = s.detach().clone()

        preds = best_s.repeat(n_rows, 1)

        loss, loss_bce = compute_soft_selection_loss(
            preds=preds,
            targets=targets_t,
            bce_loss_fn=bce_loss_fn,
        )

        total_bce += loss_bce.item() * n_rows
        total_items += n_rows
        n_qids += 1

    return {
        "bce_lower_bound": total_bce / total_items if total_items > 0 else 0.0,
        "n_items": total_items,
        "n_qids": n_qids,
    }


def compute_rank_lower_bound(
    df: pd.DataFrame,
    dimensions: list[str],
    qid_col: str = "qid",
    margin: float = 0.1,
    lr: float = 0.1,
    steps: int = 300,
):
    """
    Empirical lower bound for ranking loss.

    Assumes:
        - each row = one annotator label for the same qid
        - rank targets use:
            0 = not selected
            1 = best
            2 = second
            ...

    For each qid:
        optimize one free score vector s*
        to minimize average pairwise ranking loss
        across all annotators of that qid.

    Returns:
        dataset-level average ranking lower bound
    """

    missing = [d for d in dimensions if d not in df.columns]
    if missing:
        raise ValueError(f"Missing target columns: {missing}")
    if qid_col not in df.columns:
        raise ValueError(f"Missing qid column: {qid_col}")

    rank_loss_fn = nn.MarginRankingLoss(
        margin=margin,
        reduction="mean",
    )

    total_rank = 0.0
    total_pairs = 0
    n_qids = 0

    grouped = df.groupby(qid_col, sort=False)

    for _, g in grouped:
        y_all = g[dimensions].to_numpy(dtype=np.float32)
        n_rows, num_labels = y_all.shape

        if n_rows == 0:
            continue

        targets_t = torch.tensor(y_all, dtype=torch.float32)

        # free score vector for this qid
        s = torch.zeros(
            (1, num_labels),
            dtype=torch.float32,
            requires_grad=True,
        )

        optimizer = torch.optim.Adam([s], lr=lr)

        best_loss = None
        best_s = None

        for _ in range(steps):
            optimizer.zero_grad()

            preds = s.repeat(n_rows, 1)

            loss, loss_rank, n_pairs = compute_ranking_loss(
                preds=preds,
                targets=targets_t,
                rank_loss_fn=rank_loss_fn,
            )

            # if this qid has no valid ranking pairs, skip optimization
            if n_pairs == 0:
                best_s = s.detach().clone()
                best_loss = 0.0
                break

            loss.backward()
            optimizer.step()

            if best_loss is None or loss.item() < best_loss:
                best_loss = loss.item()
                best_s = s.detach().clone()

        preds = best_s.repeat(n_rows, 1)

        loss, loss_rank, n_pairs = compute_ranking_loss(
            preds=preds,
            targets=targets_t,
            rank_loss_fn=rank_loss_fn,
        )

        if n_pairs > 0:
            total_rank += loss_rank.item() * n_pairs
            total_pairs += n_pairs

        n_qids += 1

    return {
        "rank_lower_bound": total_rank / total_pairs if total_pairs > 0 else 0.0,
        "n_pairs": total_pairs,
        "n_qids": n_qids,
        "margin": margin,
        "lr": lr,
        "steps": steps,
    }