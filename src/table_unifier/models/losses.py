"""Функции потерь и стратегии майнинга триплетов.

Triplet Loss (из Обучение Triplet Loss.md):
  L = max(d(a, p) − d(a, n) + margin, 0)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """Обёртка над TripletMarginLoss с Евклидовой метрикой."""

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss_fn(anchor, positive, negative)


# ------------------------------------------------------------------ #
#  Онлайн-майнинг триплетов
# ------------------------------------------------------------------ #

def mine_semi_hard(
    embeddings: torch.Tensor,
    triplet_indices: torch.Tensor,
    margin: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Semi-hard mining: отбрасываем easy-триплеты.

    Args:
        embeddings:      [N_rows, D] — все эмбеддинги строк.
        triplet_indices: [T, 3] — (anchor_idx, positive_idx, negative_idx).
        margin:          отступ.

    Returns:
        (anchor_emb, positive_emb, negative_emb) — отфильтрованные триплеты.
    """
    a = embeddings[triplet_indices[:, 0]]
    p = embeddings[triplet_indices[:, 1]]
    n = embeddings[triplet_indices[:, 2]]

    d_ap = torch.sum((a - p) ** 2, dim=-1)
    d_an = torch.sum((a - n) ** 2, dim=-1)

    # Semi-hard: d(a,p) < d(a,n) < d(a,p) + margin
    mask = (d_an > d_ap) & (d_an < d_ap + margin)

    # Если semi-hard слишком мало, берём ещё hard-триплеты
    if mask.sum() < max(1, len(mask) // 4):
        hard_mask = d_an <= d_ap
        mask = mask | hard_mask

    if mask.sum() == 0:
        return a, p, n  # fallback: все триплеты

    return a[mask], p[mask], n[mask]


def online_hard_mining(
    embeddings: torch.Tensor,
    anchor_indices: torch.Tensor,
    positive_indices: torch.Tensor,
    all_negative_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Для каждого anchor находим ближайший negative (hard).

    Args:
        embeddings:           [N, D]
        anchor_indices:       [B]
        positive_indices:     [B]
        all_negative_indices: [M] — индексы всех возможных negatives.

    Returns:
        (anchor_emb, positive_emb, hardest_neg_emb)
    """
    a = embeddings[anchor_indices]         # [B, D]
    p = embeddings[positive_indices]       # [B, D]
    neg_pool = embeddings[all_negative_indices]  # [M, D]

    # Расстояния от каждого anchor ко всем negatives: [B, M]
    dists = torch.cdist(a, neg_pool, p=2)
    hardest_idx = dists.argmin(dim=1)  # [B] — индекс в neg_pool
    n = neg_pool[hardest_idx]

    return a, p, n
