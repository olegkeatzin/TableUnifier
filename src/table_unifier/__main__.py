"""
Точка входа для запуска генерации данных.

Использование:
    python -m table_unifier --config unified_experiment_config.json

Генерирует унифицированный датасет для Schema Matching и Entity Resolution.
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog='table_unifier',
        description='TableUnifier — генерация данных для SM + ER',
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Путь к JSON-конфигурации (DataGenConfig)',
    )
    args = parser.parse_args()

    from .data_generation import DataGenConfig, UnifiedDatasetGenerator

    config = DataGenConfig.load(args.config)
    generator = UnifiedDatasetGenerator(config)
    generator.generate()


if __name__ == "__main__":
    main()
