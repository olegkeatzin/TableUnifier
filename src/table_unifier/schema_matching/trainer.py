"""
Тренер для Schema Matching модели с Triplet Loss.

Пайплайн:
    1. Загрузка датасета (embeddings + class labels)
    2. Forward: raw embeddings → ProjectionHead → normalized vectors
    3. Triplet Mining: pytorch-metric-learning miners
    4. Triplet Margin Loss с L2 distance
    5. Валидация: EER, Separability, Accuracy@1

Стратегии майнинга:
    - 'semihard':  d(a,p) < d(a,n) < d(a,p) + margin
    - 'hard':      ближайший негатив и дальнейший позитив
    - 'all':       все валидные триплеты
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

from pytorch_metric_learning import miners, losses, distances

from .config import SMConfig
from .model import SchemaMatchingModel
from .dataset import SMDataset, load_and_split_dataset, create_dataloaders

logger = logging.getLogger(__name__)


class SMTrainer:
    """Тренер для Schema Matching модели.
    
    Usage:
        config = SMConfig()
        trainer = SMTrainer(config)
        history = trainer.train(dataset_path='sm_dataset/sm_dataset.json')
        trainer.save_model('sm_models/best_model.pt')
    """
    
    def __init__(
        self,
        config: SMConfig,
        device: Optional[str] = None,
    ):
        self.config = config
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"SMTrainer: device={self.device}")
        
        # Модель, лосс и майнер будут инициализированы при обучении
        self.model: Optional[SchemaMatchingModel] = None
        self.optimizer = None
        self.scheduler = None
        self.miner = None
        self.loss_fn = None
        
        # История обучения
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'train_triplets': [],
            'val_separability': [],
        }
    
    def _init_model(self, input_dim: int):
        """Инициализация модели и компонентов обучения."""
        # Обновляем input_dim из данных и пересчитываем projection_dims
        if input_dim != self.config.input_dim:
            from ..config import get_default_projection_dims
            logger.info(
                f"input_dim из данных ({input_dim}) отличается от конфига "
                f"({self.config.input_dim}), пересчитываем projection_dims"
            )
            self.config.input_dim = input_dim
            self.config.projection_dims = get_default_projection_dims(
                input_dim, self.config.output_dim
            )
            logger.info(
                f"Новые projection_dims: {self.config.projection_dims}"
            )
        
        # Модель
        self.model = SchemaMatchingModel(self.config).to(self.device)
        
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Модель: {num_params:,} параметров")
        
        # Оптимизатор
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.config.scheduler_patience,
            factor=0.5,
            min_lr=1e-6,
        )
        
        # Triplet Loss
        self.loss_fn = losses.TripletMarginLoss(
            margin=self.config.margin,
            distance=distances.LpDistance(p=2, normalize_embeddings=False),
            # Embeddings уже нормализованы в модели
        )
        
        # Miner
        mining_map = {
            'semihard': miners.TripletMarginMiner(
                margin=self.config.margin,
                type_of_triplets='semihard',
                distance=distances.LpDistance(p=2, normalize_embeddings=False),
            ),
            'hard': miners.TripletMarginMiner(
                margin=self.config.margin,
                type_of_triplets='hard',
                distance=distances.LpDistance(p=2, normalize_embeddings=False),
            ),
            'all': miners.TripletMarginMiner(
                margin=self.config.margin,
                type_of_triplets='all',
                distance=distances.LpDistance(p=2, normalize_embeddings=False),
            ),
        }
        self.miner = mining_map.get(
            self.config.mining_strategy,
            mining_map['semihard']
        )
        
        logger.info(f"Mining strategy: {self.config.mining_strategy}")
        logger.info(f"Margin: {self.config.margin}")
    
    def train(
        self,
        dataset_path: str,
        save_dir: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """Обучение модели.
        
        Args:
            dataset_path: Путь к JSON датасету
            save_dir: Директория для сохранения модели
            
        Returns:
            История обучения
        """
        save_dir = save_dir or self.config.model_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Загрузка данных
        train_ds, val_ds, test_ds = load_and_split_dataset(
            dataset_path,
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
        )
        
        train_loader, val_loader, test_loader = create_dataloaders(
            train_ds, val_ds, test_ds,
            batch_size=self.config.batch_size,
        )
        
        # Инициализация модели с правильной размерностью
        self._init_model(train_ds.embed_dim)
        
        # Сохраняем label2idx для инференса
        label2idx_path = os.path.join(save_dir, 'label2idx.json')
        with open(label2idx_path, 'w', encoding='utf-8') as f:
            json.dump(train_ds.label2idx, f, ensure_ascii=False, indent=2)
        
        # Обучение
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        logger.info(f"\nНачало обучения: {self.config.num_epochs} эпох")
        logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info("-" * 70)
        
        for epoch in range(1, self.config.num_epochs + 1):
            # Train
            train_loss, train_triplets = self._train_epoch(train_loader)
            
            # Validate
            val_loss, val_separability = self._validate(val_loader)
            
            # Scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # История
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            self.history['train_triplets'].append(train_triplets)
            self.history['val_separability'].append(val_separability)
            
            # Логирование
            logger.info(
                f"Epoch {epoch:3d}/{self.config.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Separability: {val_separability:.4f} | "
                f"Triplets: {train_triplets} | "
                f"LR: {current_lr:.2e}"
            )
            
            # Best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                
                best_path = os.path.join(save_dir, 'best_model.pt')
                self.model.save(best_path)
                logger.info(f"  → Лучшая модель сохранена (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"\nEarly stopping на эпохе {epoch} (best={best_epoch})")
                break
        
        # Загружаем лучшую модель для финальной оценки
        best_path = os.path.join(save_dir, 'best_model.pt')
        if os.path.exists(best_path):
            self.model = SchemaMatchingModel.load(best_path, device=str(self.device))
        
        # Оценка на тесте
        test_metrics = self._evaluate_test(test_loader, test_ds)
        
        # Сохраняем историю и метрики
        results = {
            'history': self.history,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'test_metrics': test_metrics,
            'config': {
                'input_dim': self.config.input_dim,
                'output_dim': self.config.output_dim,
                'projection_dims': self.config.projection_dims,
                'margin': self.config.margin,
                'mining_strategy': self.config.mining_strategy,
                'num_epochs': epoch,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
            },
            'dataset': {
                'train_samples': len(train_ds),
                'val_samples': len(val_ds),
                'test_samples': len(test_ds),
                'num_classes': train_ds.num_classes,
                'embed_dim': train_ds.embed_dim,
            },
        }
        
        results_path = os.path.join(save_dir, 'training_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"\nРезультаты сохранены: {results_path}")
        logger.info(f"Лучшая эпоха: {best_epoch}, Val Loss: {best_val_loss:.4f}")
        
        return self.history
    
    def _train_epoch(self, train_loader) -> Tuple[float, int]:
        """Одна эпоха обучения."""
        self.model.train()
        total_loss = 0.0
        total_triplets = 0
        num_batches = 0
        
        for embeddings, labels in train_loader:
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            projected = self.model(embeddings)
            
            # Mining
            hard_pairs = self.miner(projected, labels)
            
            # Loss
            loss = self.loss_fn(projected, labels, hard_pairs)
            
            if loss.item() > 0:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            total_triplets += hard_pairs[0].shape[0] if len(hard_pairs) > 0 and hard_pairs[0].shape[0] > 0 else 0
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss, total_triplets
    
    @torch.no_grad()
    def _validate(self, val_loader) -> Tuple[float, float]:
        """Валидация: loss + separability metric."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_embeddings = []
        all_labels = []
        
        for embeddings, labels in val_loader:
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)
            
            projected = self.model(embeddings)
            
            hard_pairs = self.miner(projected, labels)
            loss = self.loss_fn(projected, labels, hard_pairs)
            
            total_loss += loss.item()
            num_batches += 1
            
            all_embeddings.append(projected.cpu())
            all_labels.append(labels.cpu())
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Separability metric
        all_emb = torch.cat(all_embeddings)
        all_lab = torch.cat(all_labels)
        separability = self._compute_separability(all_emb, all_lab)
        
        return avg_loss, separability
    
    def _compute_separability(self, embeddings: torch.Tensor, labels: torch.Tensor) -> float:
        """Вычисление метрики разделимости классов.
        
        Separability = mean(inter_class_dist) / mean(intra_class_dist)
        Чем больше, тем лучше разделены классы.
        """
        unique_labels = labels.unique()
        if len(unique_labels) < 2:
            return 0.0
        
        # Центроиды классов
        centroids = {}
        intra_dists = []
        
        for label in unique_labels:
            mask = labels == label
            class_emb = embeddings[mask]
            
            if len(class_emb) < 2:
                continue
            
            centroid = class_emb.mean(dim=0)
            centroids[label.item()] = centroid
            
            # Внутриклассовое расстояние
            dists = torch.cdist(class_emb.unsqueeze(0), centroid.unsqueeze(0).unsqueeze(0))
            intra_dists.append(dists.mean().item())
        
        if len(centroids) < 2 or not intra_dists:
            return 0.0
        
        # Межклассовое расстояние
        centroid_list = torch.stack(list(centroids.values()))
        inter_dists = torch.cdist(centroid_list.unsqueeze(0), centroid_list.unsqueeze(0))
        
        # Убираем диагональ
        mask = ~torch.eye(len(centroid_list), dtype=torch.bool)
        inter_mean = inter_dists[0][mask].mean().item()
        intra_mean = np.mean(intra_dists)
        
        if intra_mean < 1e-8:
            return float('inf')
        
        return inter_mean / intra_mean
    
    @torch.no_grad()
    def _evaluate_test(self, test_loader, test_ds: SMDataset) -> Dict:
        """Оценка на тестовом наборе."""
        self.model.eval()
        
        all_embeddings = []
        all_labels = []
        
        for embeddings, labels in test_loader:
            embeddings = embeddings.to(self.device)
            projected = self.model(embeddings)
            all_embeddings.append(projected.cpu())
            all_labels.append(labels)
        
        all_emb = torch.cat(all_embeddings)
        all_lab = torch.cat(all_labels)
        
        # Separability
        separability = self._compute_separability(all_emb, all_lab)
        
        # Accuracy@1: для каждого сэмпла, ближайший сосед должен быть того же класса
        dist_matrix = torch.cdist(all_emb.unsqueeze(0), all_emb.unsqueeze(0))[0]
        dist_matrix.fill_diagonal_(float('inf'))
        
        nearest_idx = dist_matrix.argmin(dim=1)
        nearest_labels = all_lab[nearest_idx]
        accuracy_at_1 = (nearest_labels == all_lab).float().mean().item()
        
        # Intra/Inter class distances
        intra_dists = []
        inter_dists = []
        
        for i in range(len(all_lab)):
            for j in range(i + 1, min(i + 100, len(all_lab))):
                d = dist_matrix[i, j].item()
                if all_lab[i] == all_lab[j]:
                    intra_dists.append(d)
                else:
                    inter_dists.append(d)
        
        metrics = {
            'separability': separability,
            'accuracy_at_1': accuracy_at_1,
            'mean_intra_dist': np.mean(intra_dists) if intra_dists else 0.0,
            'mean_inter_dist': np.mean(inter_dists) if inter_dists else 0.0,
            'num_test_samples': len(all_lab),
            'num_classes': len(all_lab.unique()),
        }
        
        logger.info("\n" + "=" * 50)
        logger.info("ТЕСТОВЫЕ МЕТРИКИ")
        logger.info("=" * 50)
        logger.info(f"Accuracy@1:      {metrics['accuracy_at_1']:.4f}")
        logger.info(f"Separability:    {metrics['separability']:.4f}")
        logger.info(f"Intra-dist:      {metrics['mean_intra_dist']:.4f}")
        logger.info(f"Inter-dist:      {metrics['mean_inter_dist']:.4f}")
        logger.info("=" * 50)
        
        return metrics
    
    def save_model(self, filepath: str):
        """Сохранить текущую модель."""
        if self.model:
            self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Загрузить модель."""
        self.model = SchemaMatchingModel.load(filepath, device=str(self.device))
