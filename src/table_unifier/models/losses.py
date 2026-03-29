"""Функции потерь и стратегии майнинга триплетов.

Triplet Loss:
  L = max(d(a, p) − d(a, n) + margin, 0)

NT-Xent (InfoNCE):
  Для каждого якоря все остальные строки — негативы, позитив — парная строка.
  O(N) негативов на якорь при N строках в графе.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def nt_xent_loss(
    embeddings: torch.Tensor,
    pos_pairs: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """NT-Xent (InfoNCE) — memory-efficient версия.

    Вместо полной [N, N] матрицы сходств вычисляет только строки
    для anchor и positive индексов: [P, N] вместо [N, N].
    При N=70K и P=50 это 14MB вместо 20GB.

    Args:
        embeddings: [N, D], L2-нормализованы.
        pos_pairs:  [P, 2], индексы совпадающих строк.
        temperature: масштабирование сходства (0.05–0.3).

    Returns:
        Скаляр — loss.
    """
    anchors = pos_pairs[:, 0]
    positives = pos_pairs[:, 1]

    P = len(anchors)

    # [P, N] — сходства только для anchor-строк (не полная [N, N])
    anchor_emb = embeddings[anchors]                     # [P, D]
    logits = (anchor_emb @ embeddings.T) / temperature   # [P, N]
    # Убираем self-similarity: logits[i, anchors[i]] = -inf
    logits[torch.arange(P, device=logits.device), anchors] = float("-inf")
    loss = F.cross_entropy(logits, positives)

    # Симметрично: positive как anchor
    pos_emb = embeddings[positives]                        # [P, D]
    logits_rev = (pos_emb @ embeddings.T) / temperature    # [P, N]
    logits_rev[torch.arange(P, device=logits_rev.device), positives] = float("-inf")
    loss = (loss + F.cross_entropy(logits_rev, anchors)) / 2

    return loss


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
