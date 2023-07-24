import json
import os
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
from lightgbm import LGBMRegressor
from prefect import task
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

EXPERIMENT_NAME = "escooters-demand-lightgbm-hpo"

mlflow.set_tracking_uri("http://16.171.140.74:5000")
mlflow.set_experiment(EXPERIMENT_NAME)


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def cross_val_score(data, features, target_name, random_state, **params):
    cross = KFold(n_splits=4, shuffle=True, random_state=random_state)
    data.reset_index(inplace=True, drop=True)
    scores = []

    for indexes in cross.split(data):
        model = LGBMRegressor(random_state=random_state, n_jobs=3, **params)
        model.fit(data.loc[indexes[0], features], data.loc[indexes[0], target_name])
        preds = model.predict(data.loc[indexes[1], features])
        score = mean_absolute_error(data.loc[indexes[1], target_name].values, preds)

        scores.append(score)

    return np.mean(scores)


def get_best_lightgbm_params(
    data,
    features,
    target_name,
    cross_val_function,
    random_state,
    n_rounds=10,
    search_space=None,
    n_jobs=3,
    **addition_model_params,
):
    if search_space is None:
        search_space = {
            # int params
            "n_estimators": hp.quniform("n_estimators", 60, 500, 25),
            "num_leaves": hp.quniform("num_leaves", 5, 300, 5),
            "max_depth": hp.quniform("max_depth", 5, 10, 1),
            "max_bin": hp.quniform("max_bin", 5, 501, 5),
            "min_child_weight": hp.quniform("min_child_weight", 25, 1401, 25),
            "bagging_freq": hp.quniform("bagging_freq", 1, 16, 1),
            # float params
            "learning_rate": hp.quniform("learning_rate", 0.005, 0.21, 0.002),
            "colsample_bytree": hp.quniform("colsample_bytree", 0.6, 0.91, 0.025),
            "lambda_l1": hp.quniform("lambda_l1", 0.05, 1, 0.05),
            "min_split_gain": hp.quniform("min_split_gain", 0.05, 0.91, 0.025),
            "bagging_fraction": hp.quniform("bagging_fraction", 0.4, 0.91, 0.025),
        }

    def proc_params(params):
        map_params = {
            "n_estimators": lambda x: int(x),
            "num_leaves": lambda x: int(x),
            "max_depth": lambda x: int(x),
            "max_bin": lambda x: int(x),
            "min_child_weight": lambda x: int(x),
            "bagging_freq": lambda x: int(x),
        }

        for param in set(params.keys()).intersection(set(map_params.keys())):
            params[param] = map_params[param](params[param])

        return params

    def objective(params):
        params["num_threads"] = n_jobs
        params = proc_params(params)
        params["app"] = "regression"
        params["application"] = "regression"

        with mlflow.start_run():
            mlflow.log_params(params)
            cv_results = cross_val_function(
                data,
                features,
                target_name,
                random_state,
                **params,
                **addition_model_params,
            )
            mlflow.log_metric("mean-absolute-error", cv_results)

        return cv_results

    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=n_rounds,
        rstate=np.random.default_rng(random_state),
    )

    return proc_params(best)


# @click.command()
# @click.option(
#     "--data_path",
#     default=,
#     help="Location where the processed data was saved",
# )
# @click.option(
#     "--num_trials",
#     default=300,
#     help="The number of parameter evaluations for the optimizer to explore",
# )
# @click.option(
#     "--random_state",
#     default=585,
#     help="Random state",
# )
def search_params(data_path: str = "./data/processed", num_trials: int = 300, random_state: int = 585):
    df = pd.read_parquet(os.path.join(data_path, "train.parquet"))
    categorical_features = ['community', 'day_of_week', 'is_weekend']
    df[categorical_features] = df[categorical_features].astype("category")

    selected_features = [
        'community',
        'day_of_year',
        'day_of_week',
        'is_weekend',
        'week',
        'month',
        'area',
        'distance_to_center',
    ]

    mape_before = cross_val_score(
        df,
        selected_features,
        "rides_number",
        random_state,
        **{"objective": "", "num_threads": 3},
    )

    for _ in range(10):
        best_params = get_best_lightgbm_params(
            df,
            selected_features,
            "rides_number",
            cross_val_score,
            random_state,
            n_rounds=num_trials,
            **{"objective": "poisson", "n_jobs": 3},
        )
        mape_after = cross_val_score(
            df,
            selected_features,
            "rides_number",
            random_state,
            **best_params,
            **{"objective": "poisson", "num_threads": 3},
        )

        print(f"mape before: {mape_before}, mape after: {mape_after}")
        print(best_params)

        if mape_after < mape_before:
            break

    return mape_after, best_params


@task(retries=3, retry_delay_seconds=2, name="Search model hyperparameters")
def hpo():
    project_dir = Path(__file__).resolve().parents[2]
    _, best_params = search_params()  # standalone_mode=False)
    with open(os.path.join(project_dir, "./models/best_params.json"), "w", encoding="utf-8") as best_params_file:
        best_params_json = json.dumps(best_params)
        best_params_file.write(best_params_json)


if __name__ == "__main__":
    hpo()
