# CalmFruits: MVP семантического поиска

Репозиторий содержит исследование MVP query-to-item: по текстовому запросу
система возвращает top-K карточек товара. Выполнены аудит данных, подготовка
каталога и первое сравнение лексического и семантического baseline.

## Запуск исследования

1. Установите зависимости: `python3 -m pip install -r requirements.txt`.
2. Получите временные параметры S3 в уроке Практикума «Неделя 1. Постановка
   задачи проектного месяца» кнопкой «Проверить».
3. Создайте локальный `.env` по образцу `.env.example` и внесите выданные
   ключи. Этот файл игнорируется Git, не публикуйте его содержимое.
4. Зарегистрируйте kernel: `python3 -m ipykernel install --user --name
   calmfruits --display-name "CalmFruits (Python 3.12)"`.
5. Выполните тетради в чистом kernel по порядку:

   - `notebooks/01_data_audit_eda.ipynb`;
   - `notebooks/02_catalog_and_indexes.ipynb`;
   - `notebooks/03_baseline_evaluation.ipynb`.

   Для автоматического прогона второй недели используйте:

   ```bash
   jupyter nbconvert --to notebook --execute --inplace \
     notebooks/02_catalog_and_indexes.ipynb \
     --ExecutePreprocessor.kernel_name=calmfruits \
     --ExecutePreprocessor.timeout=1200

   jupyter nbconvert --to notebook --execute --inplace \
     notebooks/03_baseline_evaluation.ipynb \
     --ExecutePreprocessor.kernel_name=calmfruits \
     --ExecutePreprocessor.timeout=1200
   ```

Вторая неделя читает сырой каталог, train-разметку и список `imt_id` из
`wb_products_dedup`; тексты товаров берутся только из сырого каталога.
Тестовая разметка запрещена. Производные индексы, каталог и детальные
результаты создаются в `artifacts/`, а обзор метрик — в
`reports/baseline_metrics.csv`.

На фиксированном validation-наборе лексический baseline получил NDCG@10 0,437,
семантический MiniLM — 0,246. Детали, ограничения и гипотезы следующей недели
зафиксированы в `semantic_search_project_workbook.md`.

На третьей неделе три эксперимента подтвердили TF-IDF как validation-победителя:
короткий текст MiniLM получил NDCG@10 0,178, RRF-гибрид — 0,424, общая
линейная проекция — 0,096. Эксперимент `calmfruits_semantic_search` прошёл
проверку `check_mlflow.py`. Финальный checkpoint обработал 300
test-запросов ровно один раз: NDCG@10 0,414, MRR 0,342 и HitRate@10 0,637.

## Проверки

Локальные проверки общих функций не требуют доступа к S3:

```bash
python3 -m unittest discover -s tests -v
```

После завершения каждой недели результат проходит независимое review Astra.
