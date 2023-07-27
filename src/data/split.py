import os
from pathlib import Path
from typing import Tuple

import boto3
import pandas as pd
from prefect import task

BUCKET_NAME = "serjeeon-learning-bucket"


def split_train_test(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df.sort_values("start_day", inplace=True)
    days = sorted(df["start_day"].unique())
    days_in_test_num = round(test_size * len(days))
    test_start = days[-days_in_test_num]
    train, test = df[df["start_day"] < test_start], df[df["start_day"] >= test_start]

    return train, test


def upload_data_to_s3(filename):
    client = boto3.client("s3")
    client.upload_file(filename, BUCKET_NAME, os.path.join("escooters-demand", filename))


# @click.command()
# @click.option("--test_size", default=0.15, help="Test dataset size")
# @click.option("--path_to_interim_data", default="./data/interim", help="Path to interim data")
# @click.option(
#     "--path_to_processed_data",
#     default="./data/processed",
#     help="Path to processed data",
# )
@task(retries=3, retry_delay_seconds=2, name="Split dataset into train and test datasets")
def split_dataset(
    test_size: float = 0.15,
    path_to_interim_data: str = "data/interim",
    path_to_processed_data: str = "data/processed",
):
    features_df = pd.read_parquet(Path.joinpath(Path(path_to_interim_data), "interim_features.parquet"))

    train, test = split_train_test(features_df, test_size)
    train.to_parquet(Path.joinpath(Path(path_to_processed_data), "train.parquet"))
    test.to_parquet(Path.joinpath(Path(path_to_processed_data), "test.parquet"))
    upload_data_to_s3(Path.joinpath(Path(path_to_processed_data), "test.parquet"))


if __name__ == "__main__":
    split_dataset()
