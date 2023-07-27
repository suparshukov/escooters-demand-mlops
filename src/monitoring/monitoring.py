import random
import time
from datetime import datetime, timedelta

import boto3
import mlflow.pyfunc
import pandas as pd
import psycopg
from evidently import ColumnMapping
from evidently.metrics import (
    ColumnCorrelationsMetric,
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report
from mlflow import MlflowClient

POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "example"
TRACKING_URI = "http://16.171.140.74:5000"

BUCKET_NAME = "serjeeon-learning-bucket"
MODEL_NAME = "escooter-demand-model"
MODEL_FEATURES = [
    'community',
    'day_of_year',
    'day_of_week',
    'is_weekend',
    'week',
    'month',
    'area',
    'distance_to_center',
]

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists escooter_demand_metrics;
create table escooter_demand_metrics(
timestamp timestamp,
prediction_drift float,
num_drifted_columns integer,
share_missing_values float
)
"""


def load_model(model_name):
    stage = "Production"
    client = MlflowClient(registry_uri=TRACKING_URI)
    latest_version = client.get_latest_versions(model_name, stages=[stage])[0].source
    model = mlflow.pyfunc.load_model(model_uri=latest_version)

    return model


def read_data_from_s3(key, filename):
    client = boto3.client("s3")
    client.download_file(BUCKET_NAME, key, filename)


MODEL = load_model(MODEL_NAME)
read_data_from_s3("escooters-demand/data/reference.parquet", "reference.parquet")
REFERENCE_DATA = pd.read_parquet("reference.parquet")
read_data_from_s3("escooters-demand/data/processed/test.parquet", "test.parquet")
NEW_DATA = pd.read_parquet("test.parquet")

begin = datetime(2020, 10, 8)

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
num_features = [
    'day_of_year',
    'week',
    'month',
    'area',
    'distance_to_center',
]
categorical_features = ['community', 'day_of_week', 'is_weekend']
column_mapping = ColumnMapping(
    prediction='prediction', numerical_features=num_features, categorical_features=categorical_features, target=None
)
NEW_DATA[categorical_features] = NEW_DATA[categorical_features].astype("category")

report = Report(
    metrics=[
        ColumnDriftMetric(column_name='prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        ColumnCorrelationsMetric(column_name='prediction'),
    ]
)


def prep_db():
    with psycopg.connect(
        f"host=localhost port=5432 user={POSTGRES_USER} password={POSTGRES_PASSWORD}", autocommit=True
    ) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
        with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
            conn.execute(create_table_statement)


def calculate_metrics_postgresql(curr, i):
    current_data = NEW_DATA[
        (NEW_DATA['start_day'] >= (begin + timedelta(days=i))) & (NEW_DATA['start_day'] < (begin + timedelta(i + 1)))
    ]

    current_data['prediction'] = MODEL.predict(current_data[model_features].fillna(0))

    report.run(reference_data=REFERENCE_DATA, current_data=current_data, column_mapping=column_mapping)

    result = report.as_dict()
    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']

    print(i, prediction_drift, num_drifted_columns, share_missing_values)

    curr.execute(
        "insert into escooter_demand_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values) values (%s, %s, %s, %s)",
        (begin + timedelta(i), prediction_drift, num_drifted_columns, share_missing_values),
    )


def batch_monitoring_backfill():
    prep_db()
    last_send = datetime.now() - timedelta(seconds=10)
    with psycopg.connect(
        "host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True
    ) as conn:
        for i in range(0, 9):
            with conn.cursor() as curr:
                calculate_metrics_postgresql(curr, i)

            new_send = datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send:
                last_send = last_send + timedelta(seconds=10)


if __name__ == '__main__':
    batch_monitoring_backfill()
