# CalmFruits: MVP семантического поиска товаров

## Задача

CalmFruits проверяет, улучшает ли векторный поиск товаров по текстовому
запросу существующий поиск по ключевым словам. MVP решает задачу
query-to-item: по тексту запроса система возвращает top-K карточек товара из
каталога. Сравниваются лексический поиск (TF-IDF) и семантический поиск
(`paraphrase-multilingual-MiniLM-L12-v2`) с одинаковым контрактом
`search(query, top_k)` на одном полном каталоге оценки.

## Итоги

Каталог MVP — корневые категории «Одежда» и «Обувь», одна карточка на
`imt_id`, 3 280 карточек в каталоге оценки. Протокол: 560 development и 140
validation train-запросов без пересечения текстов, бинарный порог
`relevance >= 2`, K = 1, 3, 5, 10, NDCG по исходной шкале 0–3.

| Конфигурация (validation) | NDCG@10 | MRR | Precision@5 | Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF baseline | 0,437 | 0,376 | 0,167 | 0,529 |
| RRF-гибрид TF-IDF + MiniLM, alpha=0,25 | 0,424 | 0,378 | 0,181 | 0,532 |
| MiniLM, полный текст (семантический baseline) | 0,246 | 0,243 | 0,121 | 0,310 |
| MiniLM, короткий текст | 0,178 | 0,182 | 0,074 | 0,276 |
| MiniLM + обучаемая проекция | 0,096 | 0,120 | 0,047 | 0,164 |

По основной метрике NDCG@10 выбран TF-IDF. Единственный прогон на 300
test-запросах: NDCG@10 0,414, MRR 0,342, Recall@10 0,530, HitRate@10 0,637.
Гипотезы, разбор ошибок, ограничения, рекомендации бизнесу и дальнейшие шаги
описаны в `semantic_search_project_workbook.md`. Источники чисел:
`reports/validation_experiment_comparison.csv` и `reports/final_test_metrics.csv`.

## Структура репозитория

```text
notebooks/
  01_data_audit_eda.ipynb          аудит данных и EDA (неделя 1)
  02_catalog_and_indexes.ipynb     подготовка каталога, индексы TF-IDF и MiniLM
  03_baseline_evaluation.ipynb     golden-set, протокол оценки, baseline-метрики
  04_experiments.ipynb             три эксперимента, MLflow, выбор конфигурации
  05_final_evaluation.ipynb        единственный финальный прогон на test
src/calmfruits/                    общий код: данные, EDA, каталог, поиск,
                                   метрики, эксперименты, MLflow, final guard
tests/test_eda_helpers.py          unit-тесты общих функций
reports/                           сводные CSV/JSON метрик и выбранная конфигурация
artifacts/                         производные артефакты (не хранятся в Git)
semantic_search_project_workbook.md  рабочий документ с выводами по неделям
check_mlflow.py                    проверка MLflow-эксперимента из шаблона
context/project-month/             тексты уроков проектного месяца
requirements.txt                   зафиксированные зависимости
.env.example                       образец переменных окружения
```

Каждая тетрадь отвечает за один этап; они выполняются строго по порядку, так
как следующие читают артефакты предыдущих из `artifacts/`.

## Окружение

- Python 3.12 (тетради выполнены на 3.12.6), CPU.
- Зависимости зафиксированы в `requirements.txt`.
- Единый `SEED=42` во всех тетрадях.

## Запуск

1. Установите зависимости:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

2. Получите временные параметры S3 в уроке «Неделя 1. Постановка задачи
   проектного месяца» и параметры MLflow в уроке «Неделя 3. Улучшение
   системы, логирование в MLflow, финальные выводы» (кнопки «Проверить»).
3. Скопируйте `.env.example` в `.env` и заполните ключи S3 и MLflow. Файл
   `.env` исключён из Git, не публикуйте его.
4. Зарегистрируйте kernel:

   ```bash
   python -m ipykernel install --user --name calmfruits \
     --display-name "CalmFruits (Python 3.12)"
   ```

5. Выполните тетради по порядку в чистом kernel:

   ```bash
   for nb in 01_data_audit_eda 02_catalog_and_indexes 03_baseline_evaluation \
             04_experiments 05_final_evaluation; do
     jupyter nbconvert --to notebook --execute --inplace "notebooks/$nb.ipynb" \
       --ExecutePreprocessor.kernel_name=calmfruits \
       --ExecutePreprocessor.timeout=1800
   done
   ```

6. Проверьте MLflow-эксперимент:

   ```bash
   python check_mlflow.py calmfruits_semantic_search --env-file .env
   ```

## Данные и артефакты

Исходные данные читаются напрямую из бакета `s3-ds-source`
(`https://storage.yandexcloud.net`) и не загружаются в репозиторий:
сырой каталог товаров, `queries_synthetic_train.parquet` и
`queries_synthetic_test.parquet`. Тестовая разметка читается только в
`05_final_evaluation.ipynb`.

Тетради создают в `artifacts/` подготовленный каталог (Parquet), индекс
TF-IDF, эмбеддинги MiniLM (`.npy`), детальные результаты экспериментов и
финальный checkpoint. Эти файлы исключены из Git и воспроизводятся шагом 5.

Финальная тетрадь защищена от повторного использования test: первый запуск
сохраняет `artifacts/final/final_test_state.json`, повторные запуски загружают
сохранённый результат. В чистом клоне репозитория checkpoint отсутствует,
поэтому тетрадь выполнит test-прогон один раз.

## Проверки

Unit-тесты общих функций не требуют доступа к S3 и MLflow:

```bash
python -m unittest discover -s tests -v
```

## Streamlit

Дополнительное задание четвёртой недели (Streamlit-приложение) не
выполнялось: по критериям проекта оно бонусное и не влияет на зачёт.
