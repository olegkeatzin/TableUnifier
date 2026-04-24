1. https://www.kaggle.com/datasets/kirill57678/lamoda-products - Датасет с информацией о товарах Ламоды, в частности фото товара, фото товаров рекомендуемых к покупке с этим товаром и фото похожих товаров. Для моей задачи фото не нужны. Нужно разделить столбец About на несколько столбцов, он содержит json с примерно таким содержимым: {'Состав, %': 'Хлопок - 60%, Полиэстер - 40%', 'Сезон': 'мульти', 'Размер товара на модели': 'M INT'...
2. https://www.kaggle.com/datasets/andreevd/yandex-studcamp-24-dataset-01 - Датасет вакансий
3. https://www.kaggle.com/datasets/egorledyaev/russian-car-market-dataset?select=test.csv - Датасет содержит информацию об автомобилях, выставленных на продажу на российской автомобильной площадке. Каждая строка представляет собой отдельное объявление о продаже автомобиля с подробными характеристиками.

4. https://www.kaggle.com/datasets/fiftin/ozon-what-products-do-users-add-to-favs - In this dataset, OZON collected products (in Russian) that users most often added to their Favorites, and which have been out of stock for more than 15 days. OZON divided such products into two groups:

    Products added to Favorites the most times in the last month, which are out of stock for more than 15 days
    Products that users have most added to favorites in history and which are out of stock for more than 15 days

5. https://www.kaggle.com/datasets/rzabolotin/auto-ru-car-ads-parsed - Как и в случае датасета Ламоды, есть json столбец complectation_dict который нужно распарсить на новые столбцы
6. https://www.kaggle.com/datasets/snezhanausova/all-auto-ru-14-11-2020csv - Аналогичный 3-ему датасет, есть json столбец equipment_dict который нужно распарсить на новые столбцы
7.https://www.kaggle.com/datasets/ruslanusov/dataset-of-electronics-with-lifecycle-and-specs?select=device_dataset_with_status_15000_ru.json - По моему мнению самый важный датасет из всех, так как систему мы планируем использзовать для номенклатур подобных товаров. Используй русский вариант device_dataset_with_status_15000_ru.json. DeviceStatus 15K is a dataset containing 15,000 electronic devices (monitors, routers, scanners, projectors, IP phones, and more). Each record includes:

    📦 Model specifications
    🧾 Manufacturer, model family, release year
    🔧 Technical specs (ports, resolution, connectivity type, power, etc.)
    📊 Device status (in stock / in use)
    🌡 Climate zone of usage
    🔁 Usage intensity
    ⏳ Lifecycle data: start year, predicted and actual breakdown year
📂 Data Structure

Each object is structured like this:

{
  "device_model": "PRO-352",
  "manufacturer": "Lenovo",
  "device_type": "Projector",
  "release_year": 2013,
  "status": "in stock",
  "usage_intensity": "low",
  "climate_zone": "moderate",
  "technical_specs": {
    "brightness_lumens": 3000,
    "resolution": "Full HD"
  },
  "start_year": 2014,
  "service_life_years": 7,
  "predicted_break_year": 2021,
  "actual_break_year": 2022
}

Что нужно проанализировать?
1. Как я и говорил нужно разделить json столбцы
2. Проверить, можно ли feature engeneering сделать новые столбцы
3. Исключить столбцы не нужные для Entity Resolution и которые могут запутать LLM при разметке
4. В ДАТАСЕТАХ ДОЛЖНЫ БЫТЬ ДУБЛИКАТЫ (не полные очевидно) - чтобы собственно было что размечать LLM для Entity Resolution
