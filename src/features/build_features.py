"""Module for generating features"""
import json
import pickle
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from prefect import task
from shapely.geometry import Point, shape

JSONType = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]


def correct_formats(data: pd.DataFrame) -> pd.DataFrame:
    """
    Function corrects data formatting in rides data, such as Datetime, Float values etc.
    @param data: input pandas DataFrame
    @return: DataFrame with correct formats
    """
    data["Start Time"] = pd.to_datetime(data["Start Time"])
    data["End Time"] = pd.to_datetime(data["End Time"])

    return data


def get_dataset_to_featurize(rides_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function creates a dataset with all combinations of communities and dates
    @param rides_df: pd.DataFrame, input DataFrame with rides data
    @return: pd.DataFrame, a dataset with all combinations of communities
    and dates and target value for each combination  - the number of rides
    """
    rides_df["start_day"] = rides_df["Start Time"].dt.date
    days = list(pd.date_range(rides_df["start_day"].min(), rides_df["start_day"].max(), freq="1D"))

    rides_df.dropna(subset=["Start Community Area Number"], inplace=True)
    full_df = pd.DataFrame(
        list(product(days, rides_df["Start Community Area Number"].unique())),
        columns=["start_day", "community"],
    )
    full_df.sort_values(["start_day", "community"], inplace=True)
    full_df["community"] = full_df["community"].astype(int)

    community_rides_stat = (
        rides_df.groupby([pd.Grouper(key="Start Time", freq="d"), "Start Community Area Number"])
        .agg(rides_number=("Start Community Area Number", "count"))
        .reset_index()
    )
    community_rides_stat["start_day"] = pd.to_datetime(community_rides_stat["Start Time"].dt.date)
    community_rides_stat.columns = [
        "Start Time",
        "community",
        "rides_number",
        "start_day",
    ]

    full_df = full_df.merge(community_rides_stat, how="left", on=["start_day", "community"]).fillna(0)
    full_df = full_df[["start_day", "community", "rides_number"]]

    return full_df


def build_features_on_date(data: pd.DataFrame) -> pd.DataFrame:
    """
    Function generates features based on date
    @param data: input DataFrame
    @return: DataFrame with features on date
    """

    data["day_of_year"] = data["start_day"].dt.dayofyear
    data["day_of_week"] = data["start_day"].dt.day_of_week
    data["is_weekend"] = data["day_of_week"].isin({5, 6}).astype(int)
    data["week"] = data["start_day"].dt.isocalendar().week.astype(int)
    data["month"] = data["start_day"].dt.month.astype(int)

    return data


def get_center_lat_lon(path_to_file: Path) -> Tuple[float, float]:
    """
    Function loads the coordinates of the city center from a file
    @param path_to_file:
    @return:
    """

    with open(path_to_file, "r", encoding="utf-8") as file_with_coordinates:
        city_center_coordinates = file_with_coordinates.read()
        lat, lon = [float(x) for x in city_center_coordinates.split(",")]

    return lat, lon


def build_features_on_geodata(data: pd.DataFrame, boundaries: JSONType, lat: float, lon: float) -> pd.DataFrame:
    """
    Function generates features based on geographical data -
    communities geometry and coordinates of the city center
    @param data: input DataFrame
    @param boundaries: geometry of communities boundaries
    @param lat: latitude of the city center
    @param lon: longitude of the city center
    @return: DataFrame with features on geographical data
    """

    communities_geometry_dict = {feature["properties"]["community"]: feature["geometry"] for feature in boundaries}

    data = data[data["community_name"] != "None"]
    data["geometry"] = data["community_name"].apply(lambda x: shape(communities_geometry_dict[x]))
    data["area"] = data["geometry"].apply(lambda x: x.area)
    data["distance_to_center"] = data["geometry"].apply(lambda x: x.centroid.distance(Point(lon, lat)))

    return data


def create_community_codes_dict(data):
    community_codes_df = data[['Start Community Area Number', 'Start Community Area Name']]
    community_codes_df.dropna(inplace=True)
    community_codes_df.loc[:, 'Start Community Area Number'] = community_codes_df['Start Community Area Number'].astype(
        int
    )
    community_codes_df_dict = (
        community_codes_df[['Start Community Area Number', 'Start Community Area Name']]
        .drop_duplicates()
        .dropna()
        .set_index('Start Community Area Name')['Start Community Area Number']
        .to_dict()
    )

    return community_codes_df_dict


# @click.command()
# @click.option("--path_to_raw_data", default="./data/raw", help="Path to raw data")
# @click.option("--path_to_external_data", default="./data/external", help="Path to external data")
# @click.option("--path_to_interim_data", default="./data/interim", help="Path to interim data")
@task(retries=3, retry_delay_seconds=2, name="Build features")
def featurize(
    path_to_raw_data: str = "./data/raw",
    path_to_external_data: str = "./data/external",
    path_to_interim_data: str = "./data/interim",
    path_to_references: str = "./data/references",
):
    rides_data_filepath = Path.joinpath(Path(path_to_raw_data), "rides_data.parquet")
    input_data = pd.read_parquet(rides_data_filepath)

    clean_data = correct_formats(input_data)

    dataset_to_featurize = get_dataset_to_featurize(clean_data)
    features = build_features_on_date(dataset_to_featurize)

    with open(Path.joinpath(Path(path_to_raw_data), "boundaries.json"), "r", encoding="utf-8") as boundaries_file:
        geometry_data = json.load(boundaries_file)["features"]

    latitude, longitude = get_center_lat_lon(Path.joinpath(Path(path_to_external_data), "city_center_coordinates.txt"))

    community_codes_dict = create_community_codes_dict(clean_data)
    community_codes_inv_dict = nv_map = {v: k for k, v in community_codes_dict.items()}
    features['community_name'] = features['community'].map(community_codes_inv_dict)

    features = build_features_on_geodata(features, geometry_data, latitude, longitude)

    features_filepath = Path.joinpath(Path(path_to_interim_data), "interim_features.parquet")
    features_filepath.parent.mkdir(parents=True, exist_ok=True)

    community_codes_filepath = Path.joinpath(Path(path_to_references), "community_codes_dict.pkl")
    community_codes_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(community_codes_filepath, "wb") as community_codes_file:
        pickle.dump(community_codes_dict, community_codes_file)

    features_names = [
        "start_day",
        "community",
        "rides_number",
        "day_of_year",
        "day_of_week",
        "is_weekend",
        "week",
        "month",
        "area",
        "distance_to_center",
    ]
    features = features[features["start_day"] < datetime(2020, 10, 18)]
    features[features_names].to_parquet(features_filepath)
    print('ready')


if __name__ == "__main__":
    featurize()
