"""
Инструменты для визуализации результатов унификации
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
from .core import EmbSerialV2


def plot_similarity_matrix(reference_cols: List[EmbSerialV2], 
                           target_cols: List[EmbSerialV2],
                           figsize=(12, 10)):
    """
    Визуализация матрицы сходства между эталонными и целевыми столбцами
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    ref_embeddings = np.array([col.embedding for col in reference_cols])
    target_embeddings = np.array([col.embedding for col in target_cols])
    
    similarity_matrix = cosine_similarity(ref_embeddings, target_embeddings)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        similarity_matrix,
        xticklabels=[col.name for col in target_cols],
        yticklabels=[col.name for col in reference_cols],
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Косинусное сходство'}
    )
    plt.title('Матрица сходства столбцов', fontsize=14, fontweight='bold')
    plt.xlabel('Целевые столбцы', fontsize=12)
    plt.ylabel('Эталонные столбцы', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return plt.gcf()


def plot_matching_quality(metrics: Dict, figsize=(10, 6)):
    """
    Визуализация качества сопоставления
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # График 1: Статистика сопоставления
    categories = ['Сопоставлено', 'Не найдено', 'Лишние']
    values = [
        metrics['matched_columns'],
        metrics['total_reference_columns'] - metrics['matched_columns'],
        metrics['total_target_columns'] - metrics['matched_columns']
    ]
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    
    axes[0].bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Количество столбцов', fontsize=11)
    axes[0].set_title('Результаты сопоставления', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(values):
        axes[0].text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
    
    # График 2: Распределение сходства
    if 'detailed_mapping' in metrics:
        similarities = [m['similarity'] for m in metrics['detailed_mapping'] if m['target'] is not None]
        
        if similarities:
            axes[1].hist(similarities, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
            axes[1].axvline(metrics['average_similarity'], color='red', 
                           linestyle='--', linewidth=2, label=f"Среднее: {metrics['average_similarity']:.3f}")
            axes[1].axvline(metrics['threshold_used'], color='orange', 
                           linestyle='--', linewidth=2, label=f"Порог: {metrics['threshold_used']:.3f}")
            axes[1].set_xlabel('Косинусное сходство', fontsize=11)
            axes[1].set_ylabel('Количество пар', fontsize=11)
            axes[1].set_title('Распределение сходства', fontsize=12, fontweight='bold')
            axes[1].legend()
            axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_embedding_space(columns: List[EmbSerialV2], method='tsne', figsize=(10, 8)):
    """
    Визуализация эмбеддингов в 2D пространстве
    
    Args:
        columns: Список объектов EmbSerialV2
        method: 'tsne' или 'pca'
    """
    embeddings = np.array([col.embedding for col in columns])
    names = [col.name for col in columns]
    
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(columns)-1))
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError(f"Неизвестный метод: {method}")
    
    coords_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=figsize)
    plt.scatter(coords_2d[:, 0], coords_2d[:, 1], s=100, alpha=0.6, c=range(len(columns)), cmap='viridis')
    
    for i, name in enumerate(names):
        plt.annotate(name, (coords_2d[i, 0], coords_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8)
    
    plt.title(f'Визуализация эмбеддингов столбцов ({method.upper()})', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Компонента 1', fontsize=11)
    plt.ylabel('Компонента 2', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def generate_mapping_report(metrics: Dict) -> pd.DataFrame:
    """
    Генерация табличного отчета о сопоставлении
    """
    if 'detailed_mapping' not in metrics:
        return pd.DataFrame()
    
    report_data = []
    for mapping in metrics['detailed_mapping']:
        report_data.append({
            'Эталонный столбец': mapping['reference'],
            'Целевой столбец': mapping['target'] if mapping['target'] else '❌ НЕ НАЙДЕН',
            'Сходство': f"{mapping['similarity']:.3f}",
            'Статус': '✅ Сопоставлен' if mapping['target'] else '❌ Отсутствует'
        })
    
    df = pd.DataFrame(report_data)
    return df


def print_summary(metrics: Dict):
    """
    Красивый вывод итоговой статистики
    """
    print("\n" + "="*60)
    print("📊 ИТОГОВАЯ СТАТИСТИКА УНИФИКАЦИИ".center(60))
    print("="*60)
    
    print(f"\n🎯 Столбцы:")
    print(f"   • Эталонных:        {metrics['total_reference_columns']}")
    print(f"   • В целевой таблице: {metrics['total_target_columns']}")
    print(f"   • Сопоставлено:     {metrics['matched_columns']} "
          f"({metrics['matched_columns']/metrics['total_reference_columns']*100:.1f}%)")
    
    if metrics['matched_columns'] > 0:
        print(f"\n📈 Качество сопоставления:")
        print(f"   • Среднее сходство:  {metrics['average_similarity']:.3f}")
        print(f"   • Минимум:          {metrics['min_similarity']:.3f}")
        print(f"   • Максимум:         {metrics['max_similarity']:.3f}")
        print(f"   • Порог:            {metrics['threshold_used']:.3f}")
    
    if 'missing_columns' in metrics and metrics['missing_columns']:
        print(f"\n⚠️  Отсутствующие столбцы ({len(metrics['missing_columns'])}):")
        for col in metrics['missing_columns']:
            print(f"   • {col}")
    
    print("\n" + "="*60 + "\n")
