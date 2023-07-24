import json
import os
from datetime import datetime

import mlflow
import pandas as pd
from lightgbm import LGBMRegressor
from mlflow import MlflowClient
from prefect import task

EXPERIMENT_NAME = "escooters-demand-lightgbm-hpo"

mlflow.set_tracking_uri("http://16.171.140.74:5000")
mlflow.set_experiment(EXPERIMENT_NAME)


def train_lgbm_model(train: pd.DataFrame, model_features, categorical_features, model_params):
    (X_train, y_train) = (train[model_features], train["rides_number"])
    X_train[categorical_features] = X_train[categorical_features].astype("category")

    lgbm = LGBMRegressor(objective="poisson", **model_params)

    lgbm.fit(X_train, y_train)

    return lgbm


# @task(retries=3, retry_delay_seconds=2, name="Train a model and log it")
def train_log_model():
    train = pd.read_parquet(os.path.join("./data/processed", "train.parquet"))
    model_features = [
        'community',
        'day_of_year',
        'day_of_week',
        'is_weekend',
        'week',
        'month',
        'area',
        'distance_to_center',
    ]

    model_params_path = "./models/best_params.json"
    with open(model_params_path, "r", encoding="utf-8") as file_with_model:
        model_params = json.load(file_with_model)

    categorical_features = ['community', 'day_of_week', 'is_weekend']

    with mlflow.start_run() as run:
        model = train_lgbm_model(train, model_features, categorical_features, model_params)
        model.booster_.save_model("./models/model.txt")

        artifact_path = "model"
        model_info = mlflow.lightgbm.log_model(model, artifact_path)

        model_name = "escooter-demand-model"
        # desc = f'Best model on {datetime.now()}'
        # new_run_id = run.info.run_id
        client = MlflowClient()
        model_version = mlflow.register_model(model_uri=model_info.model_uri, name=model_name)
        print(model_version)
        # model_version = client.create_model_version(name=name,
        #                                             run_id=new_run_id,
        #                                             source=model_info.model_uri,
        #                                             description=desc)
        client.transition_model_version_stage(model_name, model_version.version, "Production")


if __name__ == "__main__":
    train_log_model()


# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#
# import seaborn as sns
# test = pd.read_parquet('../data/processed/test.parquet')
# X_test, y_test = test[model_features], test['rides_number']
# X_test[categorical_features] = X_test[categorical_features].astype("category")
# X_test["week"] = X_test["week"].astype(int)
#
# y_pred = model.predict(X_test)
#
# X_test['y_true'] = y_test
# X_test['y_pred'] = np.round(y_pred)
#
# mean_absolute_error(X_test['y_true'], X_test['y_pred'])
# mean_squared_error(X_test['y_true'], X_test['y_pred'], squared=False)
# r2_score(X_test['y_true'], X_test['y_pred'])
#
# sns.histplot(X_test[['y_pred', 'y_true']])
#
# sns.lineplot(X_test.groupby('day_of_year').agg({'y_true': 'sum', 'y_pred': 'sum'}))
#
# pass
