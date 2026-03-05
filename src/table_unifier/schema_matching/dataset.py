"""
PyTorch Dataset для Schema Matching с Triplet Loss.

Загружает датасет из JSON (сгенерированный SMDatasetGenerator),
предоставляет эмбеддинги и метки классов для triplet mining.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class SMDataset(Dataset):
    """Dataset для Schema Matching.
    
    Каждый элемент: (embedding_tensor, class_label_int)
    
    Метки (class labels) = base_name столбцов:
        'article' → 0, 'base_price' → 1, 'quantity' → 2, ...
    
    Triplet mining выполняется в trainer на уровне батча:
        - Anchor и Positive: одинаковый class_label
        - Negative: другой class_label
    """
    
    def __init__(
        self,
        data: List[Dict],
        label2idx: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            data: Список записей из JSON [{"embedding": [...], "base_name": "...", ...}]
            label2idx: Маппинг base_name → int. Если None, создается автоматически.
        """
        self.data = data
        
        # Создаём маппинг меток
        if label2idx is None:
            unique_labels = sorted(set(d['base_name'] for d in data))
            self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label2idx = label2idx
        
        self.idx2label = {v: k for k, v in self.label2idx.items()}
        self.num_classes = len(self.label2idx)
        
        # Предварительно конвертируем в тензоры
        self.embeddings = torch.tensor(
            [d['embedding'] for d in data],
            dtype=torch.float32
        )
        self.labels = torch.tensor(
            [self.label2idx[d['base_name']] for d in data],
            dtype=torch.long
        )
        
        # Метаданные для отладки
        self.column_names = [d['column_name'] for d in data]
        self.descriptions = [d.get('description', '') for d in data]
        
        logger.info(
            f"SMDataset: {len(self)} samples, {self.num_classes} classes, "
            f"embed_dim={self.embeddings.shape[1]}"
        )
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]
    
    @property
    def embed_dim(self) -> int:
        return self.embeddings.shape[1]
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Распределение по классам."""
        return Counter(d['base_name'] for d in self.data)
    
    def get_samples_per_class(self) -> Dict[int, int]:
        """Количество сэмплов для каждого класса (по int-индексу)."""
        counts = {}
        for label in self.labels.tolist():
            counts[label] = counts.get(label, 0) + 1
        return counts


def load_and_split_dataset(
    dataset_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[SMDataset, SMDataset, SMDataset]:
    """Загрузить датасет из JSON и разделить на train/val/test.
    
    Стратифицированное разделение: каждый класс пропорционально
    представлен во всех сплитах.
    
    Args:
        dataset_path: Путь к JSON файлу датасета
        train_ratio: Доля для обучения
        val_ratio: Доля для валидации
        test_ratio: Доля для тестирования
        seed: Random seed
        
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    logger.info(f"Загрузка датасета: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    logger.info(f"Всего записей: {len(all_data)}")
    
    # Группируем по классам для стратификации
    by_class: Dict[str, List[Dict]] = {}
    for entry in all_data:
        bn = entry['base_name']
        if bn not in by_class:
            by_class[bn] = []
        by_class[bn].append(entry)
    
    # Общий маппинг меток (единый для всех сплитов)
    unique_labels = sorted(by_class.keys())
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    rng = np.random.RandomState(seed)
    
    train_data, val_data, test_data = [], [], []
    
    for label, entries in by_class.items():
        rng.shuffle(entries)
        n = len(entries)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        
        train_data.extend(entries[:n_train])
        val_data.extend(entries[n_train:n_train + n_val])
        test_data.extend(entries[n_train + n_val:])
    
    # Перемешиваем внутри каждого сплита
    rng.shuffle(train_data)
    rng.shuffle(val_data)
    rng.shuffle(test_data)
    
    train_ds = SMDataset(train_data, label2idx)
    val_ds = SMDataset(val_data, label2idx)
    test_ds = SMDataset(test_data, label2idx)
    
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    logger.info(f"Classes: {len(label2idx)}, Embed dim: {train_ds.embed_dim}")
    
    return train_ds, val_ds, test_ds


def create_dataloaders(
    train_ds: SMDataset,
    val_ds: SMDataset,
    test_ds: SMDataset,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Создать DataLoader'ы для train/val/test."""
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
