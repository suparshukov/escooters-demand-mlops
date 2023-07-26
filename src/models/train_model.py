import json
import os

import boto3
import mlflow
import pandas as pd
from lightgbm import LGBMRegressor
from mlflow import MlflowClient
from prefect import task

EXPERIMENT_NAME = "escooters-demand-lightgbm-hpo"
TRACKING_URI = os.getenv("TRACKING_URI")
print(TRACKING_URI)
BUCKET_NAME = "serjeeon-learning-bucket"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def train_lgbm_model(train: pd.DataFrame, model_features, categorical_features, model_params):
    (X_train, y_train) = (train[model_features], train["rides_number"])
    X_train[categorical_features] = X_train[categorical_features].astype("category")

    lgbm = LGBMRegressor(objective="poisson", **model_params)

    lgbm.fit(X_train, y_train)

    return lgbm


def upload_reference_data(filename="./data/reference.parquet"):
    client = boto3.client("s3")
    client.upload_file(filename, BUCKET_NAME, "escooters-demand/data/reference.parquet")


@task(retry_delay_seconds=2, name="Train a model and log it")
def train_log_model():
    train = pd.read_parquet(os.path.join("./data/processed", "train.parquet"))
    val_data = pd.read_parquet(os.path.join("./data/processed", "test.parquet"))
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

        val_data[categorical_features] = val_data[categorical_features].astype("category")
        val_preds = model.predict(val_data[model_features])
        val_data['prediction'] = val_preds
        val_data.to_parquet("./data/reference.parquet")
        upload_reference_data()

        artifact_path = "model"
        model_info = mlflow.lightgbm.log_model(model, artifact_path)

        model_name = "escooter-demand-model"
        client = MlflowClient()
        model_version = mlflow.register_model(model_uri=model_info.model_uri, name=model_name)
        client.transition_model_version_stage(model_name, model_version.version, "Production")

    return model


if __name__ == "__main__":
    train_log_model()
