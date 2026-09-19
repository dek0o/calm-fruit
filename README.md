# CalmFruits: MVP семантического поиска

Репозиторий содержит исследование MVP query-to-item: по текстовому запросу
система позднее будет возвращать top-K карточек товара. На текущем этапе
выполнена неделя 1 — аудит реальных данных и фиксация решений для baseline.

## Быстрый запуск EDA

1. Установите зависимости: `python3 -m pip install -r requirements.txt`.
2. Получите временные параметры S3 в уроке Практикума «Неделя 1. Постановка
   задачи проектного месяца» кнопкой «Проверить».
3. Создайте локальный `.env` по образцу `.env.example` и внесите выданные
   ключи. Этот файл игнорируется Git, не публикуйте его содержимое.
4. Зарегистрируйте kernel: `python3 -m ipykernel install --user --name
   calmfruits --display-name "CalmFruits (Python 3.12)"`.
5. Выполните в чистом kernel
   `notebooks/01_data_audit_eda.ipynb` или командой:

   ```bash
   jupyter nbconvert --to notebook --execute --inplace \
     notebooks/01_data_audit_eda.ipynb \
     --ExecutePreprocessor.kernel_name=calmfruits \
     --ExecutePreprocessor.timeout=600
   ```

Notebook читает только `wb_products_raw_sample.parquet` и
`queries_synthetic_train.parquet` из S3. Тестовая разметка на этом этапе
запрещена. Сводные CSV и графики EDA появляются в `artifacts/eda/`; исходные
данные, секреты и тяжёлые артефакты в репозиторий не добавляются.

## Проверки

Локальные проверки общих функций не требуют доступа к S3:

```bash
python3 -m unittest discover -s tests -v
```

После завершения каждой недели результат проходит независимое review Astra.
