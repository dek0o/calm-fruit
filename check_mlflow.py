#!/usr/bin/env python3
"""
Проверка эксперимента MLflow: существует ли, есть ли минимум N успешных прогонов,
залогированы ли нужные метрики и параметры.
"""

import argparse
import os
import sys

DEFAULT_ENV_FILE = ".env"
DEFAULT_MIN_RUNS = 2
DEFAULT_METRICS = ["ndcg_at_10", "recall_at_10", "precision_at_5", "mrr"]
DEFAULT_PARAMS = ["method", "eval_split", "catalog_size", "relevance_threshold"]
ENV_URI = "MLFLOW_TRACKING_URI"
ENV_USER = "MLFLOW_TRACKING_USERNAME"
ENV_PASSWORD = "MLFLOW_TRACKING_PASSWORD"
OK, FAIL, WARN = "  [OK]  ", "  [!]   ", "  [~]   "


def load_env_file(path):
    if not os.path.isfile(path):
        return {}, "файл {} не найден".format(path)
    loaded = {}
    try:
        from dotenv import dotenv_values
        loaded = {k: v for k, v in dotenv_values(path).items() if v is not None}
    except ImportError:
        with open(path, encoding="utf-8") as source:
            for raw in source:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                if key.startswith("export "):
                    key = key[len("export "):].strip()
                loaded[key] = value.strip().strip('"').strip("'")
    for key, value in loaded.items():
        os.environ.setdefault(key, value)
    return loaded, None


def report(problems):
    print("=" * 62)
    if problems:
        print("НАЙДЕНО ЗАМЕЧАНИЙ: {}\n".format(len(problems)))
        for index, problem in enumerate(problems, 1):
            print("{}. {}".format(index, problem))
    else:
        print("ВСЁ В ПОРЯДКЕ: эксперимент, прогоны, метрики и параметры на месте.")


def main():
    parser = argparse.ArgumentParser(description="Проверка эксперимента MLflow на наличие прогонов, метрик и параметров.")
    parser.add_argument("experiment", help="название эксперимента в MLflow")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--min-runs", type=int, default=DEFAULT_MIN_RUNS)
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    parser.add_argument("--params", nargs="*", default=DEFAULT_PARAMS)
    args = parser.parse_args()
    loaded, env_error = load_env_file(args.env_file)
    if env_error:
        print("[!] {}".format(env_error))
        return 2
    tracking_uri = os.environ.get(ENV_URI)
    username = os.environ.get(ENV_USER)
    password = os.environ.get(ENV_PASSWORD)
    if not tracking_uri:
        print("[!] В {} не задан {}.".format(args.env_file, ENV_URI))
        return 2
    needs_auth = tracking_uri.startswith(("http://", "https://"))
    if needs_auth and not username or needs_auth and not password:
        print("[!] Для подключения по HTTP нужны учётные данные.")
        return 2
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("[!] MLflow не установлен.")
        return 2
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    print("Tracking URI : {}".format(tracking_uri))
    print("Эксперимент  : {}".format(args.experiment))
    print("-" * 62)
    problems = []
    try:
        experiment = client.get_experiment_by_name(args.experiment)
    except Exception as error:
        print("[!] Не удалось обратиться к MLflow: {}".format(error))
        return 2
    if experiment is None:
        print("[!] Эксперимент '{}' не найден.".format(args.experiment))
        return 1
    print("{}эксперимент найден (id={})".format(OK, experiment.experiment_id))
    runs = client.search_runs([experiment.experiment_id], max_results=1000)
    finished = [run for run in runs if run.info.status == "FINISHED"]
    if len(finished) < args.min_runs:
        problems.append("нужно минимум {} успешных прогонов, найдено {}".format(args.min_runs, len(finished)))
        print("{}успешных прогонов: {} (нужно {})".format(FAIL, len(finished), args.min_runs))
    else:
        print("{}успешных прогонов: {} (нужно {})".format(OK, len(finished), args.min_runs))
    if not finished:
        report(problems)
        return 1
    print("-" * 62)
    for run in finished:
        name = run.data.tags.get("mlflow.runName", run.info.run_id[:8])
        missing_metrics = [metric for metric in args.metrics if metric not in run.data.metrics]
        missing_params = [parameter for parameter in args.params if parameter not in run.data.params]
        if not missing_metrics and not missing_params:
            print("{}{}".format(OK, name))
            continue
        print("{}{}".format(FAIL, name))
        if missing_metrics:
            problems.append("прогон '{}': не залогированы метрики {}".format(name, ", ".join(missing_metrics)))
        if missing_params:
            problems.append("прогон '{}': не залогированы параметры {}".format(name, ", ".join(missing_params)))
    bad_names = sorted({metric for run in finished for metric in run.data.metrics if "@" in metric})
    if bad_names:
        problems.append("MLflow не разрешает '@' в именах метрик")
    methods = sorted({run.data.params.get("method") for run in finished if run.data.params.get("method")})
    if len(finished) >= args.min_runs and len(methods) < 2:
        problems.append("прогоны не различаются по параметру 'method'")
    report(problems)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
