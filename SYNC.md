# Синхронизация проекта между устройствами

Код синхронизируется через **git**, данные и модели — через **rclone** (Яндекс Диск).

## Первоначальная настройка rclone (один раз на каждом устройстве)

### Установка

- **Linux:** `sudo apt install rclone` или скачать с https://rclone.org/downloads/
- **Windows:** `winget install Rclone.Rclone` или скачать с https://rclone.org/downloads/

### Настройка Яндекс Диска

```bash
rclone config
```

1. `n` — New remote
2. Имя: `yandex`
3. Storage: `Yandex Disk`
4. Client ID / Secret — оставить пустыми
5. Auto config — `y` (если есть браузер) или `n` (headless сервер — выполнить `rclone authorize "yandex"` на машине с браузером и вставить токен)

## Ежедневный workflow

### Загрузить данные на Яндекс Диск (после работы)

```bash
git push
rclone sync data/ yandex:TableUnifier/data/ -P
rclone sync output/ yandex:TableUnifier/output/ -P
rclone copy mlflow.db yandex:TableUnifier/ -P
```

### Скачать данные с Яндекс Диска (перед работой)

```bash
git pull
rclone sync yandex:TableUnifier/data/ data/ -P
rclone sync yandex:TableUnifier/output/ output/ -P
rclone copy yandex:TableUnifier/mlflow.db . -P
```

## Что где хранится

| Что | Где | Синхронизация |
|-----|-----|---------------|
| Код, конфиги, тесты | GitHub | `git push / pull` |
| Датасеты (`data/`) | Яндекс Диск | `rclone sync` |
| Модели (`output/`) | Яндекс Диск | `rclone sync` |
| MLflow (`mlflow.db`) | Яндекс Диск | `rclone copy` |

## Примечания

- `rclone sync` передает только изменённые файлы, повторные запуски быстрые
- `-P` показывает прогресс загрузки
- `rclone copy` используется для `mlflow.db` вместо `sync`, т.к. это одиночный файл
