"""
Конфигурация модуля Entity Resolution (ER)

Содержит все параметры для:
- Построения графов (токенизация, фильтрация hub-нод)
- Модели GNN (размерности, число слоёв, attention heads)
- Обучения (learning rate, triplet margin, mining strategy)
- Генерации синтетических данных для ER
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List
import json


@dataclass
class ERConfig:
    """Конфигурация Entity Resolution на базе GNN.
    
    Архитектура:
        Двудольный граф (Row ↔ Token) с атрибутами рёбер (column embeddings).
        GNN: TransformerConv с edge_attr → метрическое обучение (Triplet Loss).
    """
    
    # ─────────────────── Построение графа ───────────────────
    min_token_length: int = 2
    """Минимальная длина токена (короткие токены слишком шумные)"""
    
    max_token_length: int = 50
    """Максимальная длина токена"""
    
    max_token_doc_freq: float = 0.8
    """Максимальная доля строк, содержащих токен (борьба с hub nodes).
    Токены, встречающиеся в > 80% строк, исключаются."""
    
    cell_separator: str = " | "
    """Разделитель значений ячеек при конкатенации строки для row embedding"""
    
    # ─────────────────── Токен-эмбеддинг (FastText) ───────────────────
    fasttext_model_path: str = "cc.ru.300.bin"
    """Путь к предобученной FastText модели (.bin формат).
    Скачать: https://fasttext.cc/docs/en/crawl-vectors.html"""
    
    token_embed_dim: int = 300
    """Размерность эмбеддинга токена (определяется FastText моделью, обычно 300)"""
    
    # ─────────────────── GNN ───────────────────
    hidden_dim: int = 256
    """Скрытая размерность GNN (все проекции приводят к этому размеру)"""
    
    edge_dim: int = 128
    """Размерность атрибутов рёбер внутри TransformerConv"""
    
    num_gnn_layers: int = 2
    """Число слоёв GNN (2-3 оптимально, больше → over-smoothing)"""
    
    num_heads: int = 4
    """Число голов внимания в TransformerConv"""
    
    dropout: float = 0.1
    """Dropout для регуляризации"""
    
    use_jumping_knowledge: bool = True
    """JumpingKnowledge: конкатенация представлений со всех слоёв.
    Борется с over-smoothing, даёт multi-scale features."""
    
    output_dim: int = 128
    """Размерность финального вектора строки (для метрического обучения)"""
    
    # ─────────────────── Обучение ───────────────────
    batch_size: int = 16
    """Размер батча (число графов-пар таблиц)"""
    
    learning_rate: float = 1e-3
    """Начальная скорость обучения"""
    
    weight_decay: float = 1e-5
    """L2-регуляризация"""
    
    num_epochs: int = 50
    """Максимальное число эпох"""
    
    margin: float = 0.3
    """Маржа для Triplet Loss: d(a, p) + margin < d(a, n)"""
    
    mining_strategy: str = "semihard"
    """Стратегия майнинга триплетов: 'hard', 'semihard', 'all'"""
    
    scheduler_patience: int = 5
    """Patience для ReduceLROnPlateau"""
    
    early_stopping_patience: int = 10
    """Patience для early stopping"""
    
    # ─────────────────── Генерация данных ER ───────────────────
    num_entity_pool: int = 500
    """Размер пула сущностей для генерации"""
    
    num_train_pairs: int = 300
    """Число пар таблиц для обучения"""
    
    num_val_pairs: int = 50
    """Число пар таблиц для валидации"""
    
    num_test_pairs: int = 50
    """Число пар таблиц для тестирования"""
    
    min_rows_per_table: int = 5
    """Минимальное число строк в таблице"""
    
    max_rows_per_table: int = 15
    """Максимальное число строк в таблице"""
    
    min_common_entities: int = 3
    """Минимальное число общих сущностей (дубликатов) в паре таблиц"""
    
    max_common_entities: int = 8
    """Максимальное число общих сущностей"""
    
    min_unique_entities: int = 1
    """Минимальное число уникальных сущностей в каждой таблице"""
    
    max_unique_entities: int = 5
    """Максимальное число уникальных сущностей"""
    
    perturbation_prob: float = 0.3
    """Вероятность пертурбации значения ячейки (опечатки, форматирование)"""
    
    missing_value_prob: float = 0.1
    """Вероятность пропуска значения (NaN)"""
    
    # ─────────────────── Ollama ───────────────────
    ollama_host: str = "http://127.0.0.1:11434"
    embedding_model: str = "qwen3-embedding:8b"
    llm_model: str = "qwen3.5:9b"
    embedding_batch_size: int = 10
    """Размер батча для Ollama embed API"""
    
    # ─────────────────── Пути ───────────────────
    output_dir: str = "er_dataset"
    """Корневая директория для данных ER"""
    
    graphs_dir: str = "graphs"
    """Поддиректория для сохранённых графов (.pt)"""

    sm_model_path: str = "sm_models/best_model.pt"
    """Path to trained SM model for column embeddings (triplet loss projection)"""
    
    def save(self, filepath: str):
        """Сохранить конфигурацию в JSON"""
        data = asdict(self)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ERConfig':
        """Загрузить конфигурацию из JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Убираем поля, которых нет в датаклассе (forward compat)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
    
    def __repr__(self):
        lines = [f"ERConfig("]
        for k, v in asdict(self).items():
            lines.append(f"  {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)
