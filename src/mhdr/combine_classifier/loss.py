import torch
import torch.nn as nn


def build_selection_targets(y: torch.Tensor) -> torch.Tensor:
    """
    Convert rank targets into selection targets.

    rank convention:
        smaller positive number = higher rank
        0 = not selected
    """
    return (y > 0).float()


def pairwise_ranking_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    rank_loss_fn: nn.MarginRankingLoss,
):
    """
    Pairwise ranking only among positive labels.

    Convention:
        smaller target value = higher rank
        0 = not selected
    """

    losses = []
    batch_size = targets.shape[0]

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

                if y[i] < y[j]:
                    hi, lo = i, j
                else:
                    hi, lo = j, i

                target = torch.tensor([1.0], device=preds.device)

                losses.append(
                    rank_loss_fn(
                        s[hi].unsqueeze(0),
                        s[lo].unsqueeze(0),
                        target,
                    )
                )

    if len(losses) == 0:
        return torch.tensor(0.0, device=preds.device), 0

    return torch.stack(losses).mean(), len(losses)


def compute_total_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    bce_loss_fn,
    rank_loss_fn,
    alpha: float,
    beta: float,
):
    selection_targets = build_selection_targets(targets)

    loss_bce = bce_loss_fn(preds, selection_targets)

    loss_rank, n_pairs = pairwise_ranking_loss(
        preds,
        targets,
        rank_loss_fn,
    )

    total_loss = alpha * loss_bce + beta * loss_rank

    return total_loss, loss_bce, loss_rank, n_pairs