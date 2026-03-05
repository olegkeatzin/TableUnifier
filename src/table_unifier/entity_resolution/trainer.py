"""
Тренер для Entity Resolution GNN с Metric Learning.

Пайплайн обучения:
    1. Загрузка предпостроенных графов (ERGraphDataset)
    2. Forward pass через GNN → L2-нормированные row embeddings
    3. Triplet Mining: выбор информативных (anchor, positive, negative)
    4. Triplet Margin Loss: d(a,p) + margin < d(a,n)
    5. Валидация: Precision@K, Recall, F1 для поиска дубликатов

Стратегии майнинга триплетов:
    - 'semihard': полу-жёсткие негативы (d(a,p) < d(a,n) < d(a,p) + margin)
      → хороший баланс скорости и качества обучения
    - 'hard': самые жёсткие негативы (ближайший негатив)
      → быстрая сходимость, но может быть нестабильно
    - 'all': все валидные триплеты
      → медленно, но самое стабильное обучение
"""

import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Optional, List, Tuple
from pathlib import Path

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import DataLoader as PyGDataLoader
except ImportError:
    raise ImportError("PyTorch Geometric не установлен! pip install torch-geometric")

try:
    from pytorch_metric_learning import miners, losses, distances
except ImportError:
    raise ImportError(
        "pytorch-metric-learning не установлен! "
        "pip install pytorch-metric-learning"
    )

from .config import ERConfig
from .gnn_model import EntityResolutionGNN
from .er_dataset import ERGraphDataset, er_collate_fn

logger = logging.getLogger(__name__)


class ERTrainer:
    """Тренер для Entity Resolution GNN.
    
    Usage:
        config = ERConfig()
        trainer = ERTrainer(config, row_input_dim=300, col_embed_dim=256)
        
        history = trainer.train(
            train_dir='er_dataset/graphs/train',
            val_dir='er_dataset/graphs/val',
        )
        
        trainer.save_model('er_model.pt')
    """
    
    def __init__(
        self,
        config: ERConfig,
        row_input_dim: int,
        col_embed_dim: int,
        device: Optional[str] = None,
    ):
        """
        Args:
            config: Конфигурация ER
            row_input_dim: Размерность row embeddings (auto-detect из графа)
            col_embed_dim: Размерность column embeddings (edge_attr dim)
            device: 'cuda', 'cpu', или None (авто)
        """
        self.config = config
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"ERTrainer: device={self.device}")
        
        # Модель
        self.model = EntityResolutionGNN(
            row_input_dim=row_input_dim,
            col_embed_dim=col_embed_dim,
            config=config,
        ).to(self.device)
        
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Модель: {num_params:,} параметров")
        
        # Оптимизатор
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler: снижаем LR когда валидация перестаёт улучшаться
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=config.scheduler_patience,
            factor=0.5,
            min_lr=1e-6,
        )
        
        # Loss: Triplet Margin Loss
        self.loss_fn = losses.TripletMarginLoss(
            margin=config.margin,
            distance=distances.CosineSimilarity(),
        )
        
        # Miner: стратегия выбора триплетов
        self._setup_miner(config.mining_strategy)
        
        # История обучения
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rate': [],
        }
    
    def _setup_miner(self, strategy: str):
        """Настроить triplet miner."""
        if strategy == 'hard':
            self.miner = miners.TripletMarginMiner(
                margin=self.config.margin,
                type_of_triplets='hard',
            )
        elif strategy == 'semihard':
            self.miner = miners.TripletMarginMiner(
                margin=self.config.margin,
                type_of_triplets='semihard',
            )
        elif strategy == 'all':
            self.miner = miners.TripletMarginMiner(
                margin=self.config.margin,
                type_of_triplets='all',
            )
        else:
            raise ValueError(f"Неизвестная стратегия: {strategy}. Используйте 'hard', 'semihard', 'all'.")
        
        logger.info(f"Triplet mining strategy: {strategy}")
    
    def train(
        self,
        train_dir: str,
        val_dir: str,
        num_epochs: int = None,
        save_dir: str = None,
    ) -> Dict:
        """Основной цикл обучения.
        
        Args:
            train_dir: Директория с .pt графами для обучения
            val_dir: Директория с .pt графами для валидации
            num_epochs: Число эпох (из конфига если None)
            save_dir: Директория для сохранения моделей и логов
            
        Returns:
            История обучения (dict)
        """
        num_epochs = num_epochs or self.config.num_epochs
        
        # Даталоадеры
        train_dataset = ERGraphDataset(train_dir)
        val_dataset = ERGraphDataset(val_dir)
        
        train_loader = PyGDataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=er_collate_fn,
        )
        val_loader = PyGDataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=er_collate_fn,
        )
        
        logger.info(f"Train: {len(train_dataset)} графов, Val: {len(val_dataset)} графов")
        logger.info(f"Batch size: {self.config.batch_size}, Epochs: {num_epochs}")
        
        # Подготовка сохранения
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # ── Обучение ──
            train_loss = self._train_epoch(train_loader)
            
            # ── Валидация ──
            val_metrics = self._validate(val_loader)
            val_loss = val_metrics['loss']
            
            # ── Scheduler ──
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # ── Логирование ──
            elapsed = time.time() - start_time
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_precision'].append(val_metrics.get('precision', 0))
            self.history['val_recall'].append(val_metrics.get('recall', 0))
            self.history['val_f1'].append(val_metrics.get('f1', 0))
            self.history['learning_rate'].append(current_lr)
            
            logger.info(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"P/R/F1: {val_metrics.get('precision', 0):.3f}/"
                f"{val_metrics.get('recall', 0):.3f}/"
                f"{val_metrics.get('f1', 0):.3f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )
            
            # ── Early stopping + сохранение лучшей модели ──
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if save_dir:
                    self.save_model(os.path.join(save_dir, 'best_model.pt'))
                    logger.info(f"  -> Лучшая модель сохранена (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(
                        f"Early stopping на эпохе {epoch} "
                        f"(patience={self.config.early_stopping_patience})"
                    )
                    break
        
        # Сохраняем историю
        if save_dir:
            history_path = os.path.join(save_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
        
        return self.history
    
    def _train_epoch(self, loader) -> float:
        """Одна эпоха обучения."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in loader:
            batch = batch.to(self.device)
            
            # Forward
            embeddings = self.model(batch)  # [N_rows, D_output]
            labels = batch['row'].entity_label  # [N_rows]
            
            # Проверяем, есть ли хотя бы 2 экземпляра одной метки (для триплетов)
            unique, counts = labels.unique(return_counts=True)
            has_duplicates = (counts > 1).any()
            
            if not has_duplicates:
                continue  # Нет дубликатов в батче — пропускаем
            
            # Mining: выбор информативных триплетов
            hard_pairs = self.miner(embeddings, labels)
            
            if hard_pairs[0].shape[0] == 0:
                continue  # Miner не нашёл триплетов
            
            # Loss
            loss = self.loss_fn(embeddings, labels, hard_pairs)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def _validate(self, loader) -> Dict[str, float]:
        """Валидация: loss + метрики поиска дубликатов."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_embeddings = []
        all_labels = []
        all_table_ids = []
        
        for batch in loader:
            batch = batch.to(self.device)
            
            embeddings = self.model(batch)
            labels = batch['row'].entity_label
            table_ids = batch['row'].table_id
            
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            all_table_ids.append(table_ids.cpu())
            
            # Loss
            unique, counts = labels.unique(return_counts=True)
            has_duplicates = (counts > 1).any()
            
            if has_duplicates:
                hard_pairs = self.miner(embeddings, labels)
                if hard_pairs[0].shape[0] > 0:
                    loss = self.loss_fn(embeddings, labels, hard_pairs)
                    total_loss += loss.item()
                    num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Вычисляем P/R/F1 для поиска дубликатов
        metrics = self._compute_retrieval_metrics(
            torch.cat(all_embeddings),
            torch.cat(all_labels),
            torch.cat(all_table_ids),
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def _compute_retrieval_metrics(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        table_ids: torch.Tensor,
        threshold: float = 0.7,
    ) -> Dict[str, float]:
        """Вычислить Precision, Recall, F1 для поиска дубликатов.
        
        Для каждой строки из таблицы A ищем ближайшую в таблице B.
        Если косинусное сходство > threshold и метки совпадают → TP.
        """
        mask_a = table_ids == 0
        mask_b = table_ids == 1
        
        emb_a = embeddings[mask_a]  # [N_a, D]
        emb_b = embeddings[mask_b]  # [N_b, D]
        labels_a = labels[mask_a]
        labels_b = labels[mask_b]
        
        if emb_a.shape[0] == 0 or emb_b.shape[0] == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Cosine similarity matrix: [N_a, N_b]
        sim_matrix = torch.mm(emb_a, emb_b.t())
        
        # Для каждой строки A находим лучшее совпадение в B
        best_sims, best_indices = sim_matrix.max(dim=1)
        
        # Ground truth: какие строки A имеют дубликаты в B
        gt_pairs = set()
        for i, label_a in enumerate(labels_a):
            for j, label_b in enumerate(labels_b):
                if label_a == label_b:
                    gt_pairs.add((i, j.item() if isinstance(j, torch.Tensor) else j))
        
        gt_a_has_dup = {pair[0] for pair in gt_pairs}
        
        tp = 0
        fp = 0
        fn = 0
        
        for i in range(emb_a.shape[0]):
            predicted_match = best_sims[i] > threshold
            predicted_j = best_indices[i].item()
            
            if i in gt_a_has_dup:
                # Эта строка имеет дубликат
                if predicted_match and (i, predicted_j) in gt_pairs:
                    tp += 1
                else:
                    fn += 1
            else:
                if predicted_match:
                    fp += 1
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def save_model(self, filepath: str):
        """Сохранить модель + конфигурацию."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'history': self.history,
        }
        torch.save(checkpoint, filepath)
    
    def load_model(self, filepath: str):
        """Загрузить модель из чекпоинта."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        logger.info(f"Модель загружена из {filepath}")
