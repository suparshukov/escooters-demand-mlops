import click
import pandas as pd
from pathlib import Path
from typing import Tuple


def split_train_test(
    df: pd.DataFrame, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df.sort_values("start_day", inplace=True)
    days = sorted(df["start_day"].unique())
    days_in_test_num = round(test_size * len(days))
    test_start = days[-days_in_test_num]
    train, test = df[df["start_day"] < test_start], df[df["start_day"] >= test_start]
    # len(train), len(test), train["start_day"].nunique(), test["start_day"].nunique()

    return train, test


@click.command()
@click.option("--test_size", default=0.2, help="Test dataset size")
@click.option(
    "--path_to_interim_data", default="./data/interim", help="Path to interim data"
)
@click.option(
    "--path_to_processed_data",
    default="./data/processed",
    help="Path to processed data",
)
def main(test_size: float, path_to_interim_data: str, path_to_processed_data: str):

    features_df = pd.read_parquet(
        Path.joinpath(Path(path_to_interim_data), "interim_features.parquet")
    )

    train, test = split_train_test(features_df, test_size)
    train.to_parquet(Path.joinpath(Path(path_to_processed_data), "train.parquet"))
    test.to_parquet(Path.joinpath(Path(path_to_processed_data), "test.parquet"))


if __name__ == "__main__":
    main()
