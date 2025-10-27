"""
Утилиты для работы с датасетом
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


class DatasetAnalyzer:
    """Анализатор датасета"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.data = self.load_dataset()
    
    def load_dataset(self) -> List[Dict]:
        """Загрузка датасета"""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_statistics(self) -> Dict:
        """Получить статистику датасета"""
        base_names = [entry['column_base_name'] for entry in self.data]
        column_names = [entry['column_name'] for entry in self.data]
        data_types = [entry.get('data_type', 'unknown') for entry in self.data]
        from_cache = [entry.get('from_cache', False) for entry in self.data]
        
        stats = {
            'total_entries': len(self.data),
            'unique_base_names': len(set(base_names)),
            'unique_column_names': len(set(column_names)),
            'base_name_distribution': Counter(base_names),
            'data_type_distribution': Counter(data_types),
            'cached_entries': sum(from_cache),
            'cache_rate': sum(from_cache) / len(self.data) if self.data else 0,
            'tables': len(set(entry['table_path'] for entry in self.data))
        }
        
        return stats
    
    def print_summary(self):
        """Красивый вывод статистики"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("СТАТИСТИКА ДАТАСЕТА")
        print("="*60)
        print(f"Всего записей:           {stats['total_entries']}")
        print(f"Уникальных базовых имен: {stats['unique_base_names']}")
        print(f"Уникальных названий:     {stats['unique_column_names']}")
        print(f"Таблиц:                  {stats['tables']}")
        print(f"Из кэша:                 {stats['cached_entries']} ({stats['cache_rate']*100:.1f}%)")
        
        print(f"\nРаспределение по типам столбцов:")
        for base_name, count in stats['base_name_distribution'].most_common():
            print(f"  {base_name:30s}: {count:4d}")
        
        print(f"\nРаспределение по типам данных:")
        for dtype, count in stats['data_type_distribution'].most_common():
            print(f"  {dtype:20s}: {count:4d}")
        
        print("="*60 + "\n")
    
    def plot_distributions(self, save_path: str = None):
        """Визуализация распределений"""
        stats = self.get_statistics()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # График 1: Распределение по базовым именам
        base_names = dict(stats['base_name_distribution'].most_common(10))
        axes[0].barh(list(base_names.keys()), list(base_names.values()), color='steelblue')
        axes[0].set_xlabel('Количество', fontsize=11)
        axes[0].set_title('Топ-10 столбцов по частоте', fontsize=12, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # График 2: Распределение по типам данных
        dtypes = stats['data_type_distribution']
        axes[1].pie(dtypes.values(), labels=dtypes.keys(), autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Распределение типов данных', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"График сохранен: {save_path}")
        
        return fig
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Экспорт датасета в DataFrame (без эмбеддингов)"""
        records = []
        for entry in self.data:
            record = {
                'column_name': entry['column_name'],
                'column_base_name': entry['column_base_name'],
                'description': entry['description'],
                'table_path': entry['table_path'],
                'data_type': entry.get('data_type', 'unknown'),
                'from_cache': entry.get('from_cache', False),
                'content_sample': str(entry['content'][:3])
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def get_embeddings_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Получить матрицу эмбеддингов и список имен"""
        embeddings = np.array([entry['embedding'] for entry in self.data])
        names = [f"{entry['column_base_name']} ({entry['column_name']})" for entry in self.data]
        return embeddings, names
    
    def find_similar_columns(self, column_name: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Найти похожие столбцы по эмбеддингам"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Найти эмбеддинг целевого столбца
        target_entry = next((e for e in self.data if e['column_name'] == column_name), None)
        if not target_entry:
            return []
        
        target_emb = np.array(target_entry['embedding']).reshape(1, -1)
        
        # Вычислить сходство со всеми
        similarities = []
        for entry in self.data:
            if entry['column_name'] == column_name:
                continue
            
            emb = np.array(entry['embedding']).reshape(1, -1)
            sim = cosine_similarity(target_emb, emb)[0][0]
            similarities.append((entry['column_name'], float(sim)))
        
        # Сортировать и вернуть топ-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def validate_dataset(self) -> Dict:
        """Валидация датасета"""
        issues = {
            'missing_embeddings': [],
            'invalid_embeddings': [],
            'missing_descriptions': [],
            'duplicate_entries': []
        }
        
        seen = set()
        for i, entry in enumerate(self.data):
            # Проверка эмбеддингов
            if 'embedding' not in entry:
                issues['missing_embeddings'].append(i)
            elif not isinstance(entry['embedding'], list) or len(entry['embedding']) == 0:
                issues['invalid_embeddings'].append(i)
            
            # Проверка описаний
            if 'description' not in entry or not entry['description']:
                issues['missing_descriptions'].append(i)
            
            # Проверка дубликатов
            key = (entry['column_name'], entry['table_path'])
            if key in seen:
                issues['duplicate_entries'].append(i)
            seen.add(key)
        
        return issues


class DatasetSplitter:
    """Разделение датасета на train/val/test"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.data = self.load_dataset()
    
    def load_dataset(self) -> List[Dict]:
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def split_by_tables(self, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                       test_ratio: float = 0.15, random_seed: int = 42) -> Dict:
        """
        Разделение по таблицам (чтобы данные из одной таблицы не попали в разные сеты)
        """
        import random
        random.seed(random_seed)
        
        # Группировка по таблицам
        tables = {}
        for entry in self.data:
            table = entry['table_path']
            if table not in tables:
                tables[table] = []
            tables[table].append(entry)
        
        # Перемешивание таблиц
        table_names = list(tables.keys())
        random.shuffle(table_names)
        
        # Разделение
        n_train = int(len(table_names) * train_ratio)
        n_val = int(len(table_names) * val_ratio)
        
        train_tables = table_names[:n_train]
        val_tables = table_names[n_train:n_train + n_val]
        test_tables = table_names[n_train + n_val:]
        
        # Сбор данных
        splits = {
            'train': [entry for table in train_tables for entry in tables[table]],
            'val': [entry for table in val_tables for entry in tables[table]],
            'test': [entry for table in test_tables for entry in tables[table]]
        }
        
        return splits
    
    def save_splits(self, splits: Dict, output_dir: str = 'dataset_splits'):
        """Сохранение разделений"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in splits.items():
            output_file = Path(output_dir) / f'{split_name}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            
            print(f"Сохранено {len(split_data)} записей в {output_file}")


def convert_to_csv(dataset_path: str, output_path: str):
    """Конвертация датасета в CSV (без эмбеддингов)"""
    analyzer = DatasetAnalyzer(dataset_path)
    df = analyzer.export_to_dataframe()
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Датасет экспортирован в {output_path}")


def merge_datasets(dataset_paths: List[str], output_path: str):
    """Объединение нескольких датасетов"""
    all_data = []
    for path in dataset_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.extend(data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"Объединено {len(all_data)} записей из {len(dataset_paths)} датасетов")
    print(f"Сохранено в {output_path}")


# --- ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ ---

if __name__ == "__main__":
    # Анализ датасета
    analyzer = DatasetAnalyzer("dataset.json")
    analyzer.print_summary()
    analyzer.plot_distributions("dataset_distribution.png")
    
    # Экспорт в CSV
    df = analyzer.export_to_dataframe()
    df.to_csv("dataset_summary.csv", index=False)
    print("Экспортировано в dataset_summary.csv")
    
    # Валидация
    issues = analyzer.validate_dataset()
    print("\nРезультаты валидации:")
    for issue_type, issue_list in issues.items():
        if issue_list:
            print(f"  {issue_type}: {len(issue_list)} проблем")
    
    # Поиск похожих столбцов
    similar = analyzer.find_similar_columns("Артикул", top_k=5)
    print("\nПохожие столбцы на 'Артикул':")
    for name, similarity in similar:
        print(f"  {name}: {similarity:.3f}")
    
    # Разделение на train/val/test
    splitter = DatasetSplitter("dataset.json")
    splits = splitter.split_by_tables(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    splitter.save_splits(splits)
    
    print("\nСтатистика разделения:")
    for split_name, split_data in splits.items():
        print(f"  {split_name}: {len(split_data)} записей")
