import torch
import torch.nn as nn


def build_selection_targets(y: torch.Tensor) -> torch.Tensor:
    """
    Convert rank targets into binary selection targets.

    Convention:
        0 = not selected
        >0 = selected
    """
    return (y > 0).float()


def compute_selection_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    bce_loss_fn: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Hard-label selection loss.
    targets are rank labels; convert to binary 0/1.
    """
    selection_targets = build_selection_targets(targets)
    loss_bce = bce_loss_fn(preds, selection_targets)
    return loss_bce, loss_bce


def compute_soft_selection_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    bce_loss_fn: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Soft-label selection loss.
    targets are already soft probabilities in [0, 1].
    """
    loss_bce = bce_loss_fn(preds, targets)
    return loss_bce, loss_bce


def pairwise_ranking_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    rank_loss_fn: nn.MarginRankingLoss,
) -> tuple[torch.Tensor, int]:
    """
    Ranking-only loss among ground-truth positive labels.

    Convention:
        smaller positive target value = higher rank
        0 = not selected

    Example:
        1 > 2 > 3 > 0
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


def compute_ranking_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    rank_loss_fn: nn.MarginRankingLoss,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Ranking-only wrapper.
    """
    loss_rank, n_pairs = pairwise_ranking_loss(
        preds=preds,
        targets=targets,
        rank_loss_fn=rank_loss_fn,
    )
    return loss_rank, loss_rank, n_pairs