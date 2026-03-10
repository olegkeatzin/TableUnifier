"""Тесты для models/losses.py."""

import torch
import pytest

from table_unifier.models.losses import TripletLoss, mine_semi_hard, online_hard_mining


class TestTripletLoss:
    def test_zero_loss_when_positive_closer(self):
        loss_fn = TripletLoss(margin=0.3)
        anchor = torch.tensor([[0.0, 0.0]])
        positive = torch.tensor([[0.1, 0.0]])   # distance = 0.1
        negative = torch.tensor([[10.0, 0.0]])   # distance = 10.0
        loss = loss_fn(anchor, positive, negative)
        assert loss.item() == 0.0  # d(a,p) - d(a,n) + margin < 0

    def test_nonzero_loss_when_negative_closer(self):
        loss_fn = TripletLoss(margin=0.3)
        anchor = torch.tensor([[0.0, 0.0]])
        positive = torch.tensor([[5.0, 0.0]])
        negative = torch.tensor([[0.1, 0.0]])
        loss = loss_fn(anchor, positive, negative)
        assert loss.item() > 0

    def test_batch(self):
        loss_fn = TripletLoss(margin=0.5)
        a = torch.randn(8, 16)
        p = torch.randn(8, 16)
        n = torch.randn(8, 16)
        loss = loss_fn(a, p, n)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_gradient(self):
        loss_fn = TripletLoss(margin=0.3)
        a = torch.randn(4, 8, requires_grad=True)
        p = torch.randn(4, 8)
        n = torch.randn(4, 8)
        loss = loss_fn(a, p, n)
        loss.backward()
        assert a.grad is not None


class TestMineSemiHard:
    def test_returns_tensors(self):
        embeddings = torch.randn(10, 16)
        triplets = torch.tensor([
            [0, 5, 8],
            [1, 6, 9],
            [2, 7, 3],
        ])
        a, p, n = mine_semi_hard(embeddings, triplets, margin=0.3)
        assert a.shape[0] == p.shape[0] == n.shape[0]
        assert a.shape[1] == 16

    def test_fallback_all_when_empty(self):
        # Все easy — d(a,n) >> d(a,p) + margin
        embeddings = torch.zeros(10, 4)
        embeddings[0] = torch.tensor([0.0, 0.0, 0.0, 0.0])
        embeddings[5] = torch.tensor([0.01, 0.0, 0.0, 0.0])  # positive very close
        embeddings[8] = torch.tensor([100.0, 100.0, 100.0, 100.0])  # negative very far
        triplets = torch.tensor([[0, 5, 8]])
        a, p, n = mine_semi_hard(embeddings, triplets, margin=0.01)
        # Должен вернуть fallback (все), т.к. semi_hard+hard = 0
        assert a.shape[0] >= 1


class TestOnlineHardMining:
    def test_finds_hardest_negative(self):
        embeddings = torch.tensor([
            [0.0, 0.0],  # 0: anchor
            [0.1, 0.0],  # 1: positive
            [0.2, 0.0],  # 2: closer neg
            [10.0, 0.0], # 3: far neg
        ])
        a, p, n = online_hard_mining(
            embeddings,
            anchor_indices=torch.tensor([0]),
            positive_indices=torch.tensor([1]),
            all_negative_indices=torch.tensor([2, 3]),
        )
        # Hardest negative is idx 2 (closest to anchor)
        torch.testing.assert_close(n, embeddings[2:3])

    def test_batch(self):
        emb = torch.randn(20, 8)
        a, p, n = online_hard_mining(
            emb,
            anchor_indices=torch.tensor([0, 1, 2]),
            positive_indices=torch.tensor([3, 4, 5]),
            all_negative_indices=torch.tensor([6, 7, 8, 9, 10]),
        )
        assert a.shape == (3, 8)
        assert p.shape == (3, 8)
        assert n.shape == (3, 8)
