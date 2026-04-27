#!/bin/bash

# 1. Имя вашего файла ноутбука
NOTEBOOK="/home/user/Рабочий стол/TableUnifier/experiments/18_ga_hdbscan_bge_m3_analysis.ipynb"

# 2. Выполняем ноутбук и сохраняем результаты прямо в него (--inplace)
# --ExecutePreprocessor.timeout=600 увеличивает время ожидания (если код долгий)
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$NOTEBOOK"

# 3. Добавляем изменения в Git
git add "$NOTEBOOK"

# 4. Делаем коммит и пушим
git commit -m '18'
git push


jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 /home/user/Рабочий стол/TableUnifier/experiments/18_ga_hdbscan_bge_m3_analysis.ipynb && git add /home/user/Рабочий стол/TableUnifier/experiments/18_ga_hdbscan_bge_m3_analysis.ipynb && git commit -m '18' && git push