"""Module providing functions for loading raw data"""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import requests
from prefect import task

JSONType = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]


def load_rides_data(
    url_download_file_from: str, path_to_save_raw_data: Path, return_data: bool = False
) -> Optional[pd.DataFrame]:
    """
    Downloads raw rides data from url into raw data local directory (../data/raw/).
    @param url_download_file_from: str, the address of the page
    for downloading rides data
    @param path_to_save_raw_data: Path, the path
    for saving the rides data file
    @param return_data: bool, a flag that determines
    whether the rides dataset should be returned
    @return: pandas DataFrame, the rides dataset
    """

    if path_to_save_raw_data.exists():
        print("raw rides data already exist. skipping downloading")
    else:
        print("downloading raw rides data")
        response = requests.get(url_download_file_from, allow_redirects=True, timeout=3)
        temp_csv_file = os.path.splitext(str(path_to_save_raw_data))[0] + ".csv"
        path_to_save_raw_data.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_csv_file, "wb") as temp_file:
            temp_file.write(response.content)
        data = pd.read_csv(temp_csv_file, sep=";")
        data.dropna(subset=["Start Community Area Name"], inplace=True)
        data.loc[:, "Start Community Area Number"] = data["Start Community Area Number"].astype(int)
        data.to_parquet(path_to_save_raw_data)
        print(f"raw rides data saved to {path_to_save_raw_data}")
        os.remove(temp_csv_file)

    data = pd.read_parquet(path_to_save_raw_data)

    if return_data:
        return data
    return None


def load_boundaries_data(
    url_download_file_from: str, path_to_save_raw_data: Path, return_data: bool = False
) -> Optional[JSONType]:
    """
    Downloads geodata on city's districts boundaries from url
    into raw data local directory (../data/raw/).
    @param url_download_file_from: str, the address of the page
    for downloading the districts boundaries data
    @param path_to_save_raw_data: Path, the path
    for saving the rides data file
    @param return_data: bool, a flag that determines
    whether the districts boundaries data should be returned
    @return: json: the districts boundaries data
    """

    if path_to_save_raw_data.exists():
        print("boundaries data already exist. skipping downloading")
        with open(path_to_save_raw_data, "r", encoding="utf-8") as file_with_boundaries:
            communities_boundaries_json = json.load(file_with_boundaries)
    else:
        print("downloading boundaries data")
        response = requests.get(url_download_file_from, allow_redirects=True, timeout=3)
        communities_boundaries_json = json.loads(response.text)
        path_to_save_raw_data.parent.mkdir(parents=True, exist_ok=True)
        with open(path_to_save_raw_data, "w", encoding="utf-8") as file_with_boundaries:
            json.dump(communities_boundaries_json, file_with_boundaries)
        print(f"boundaries data saved to {path_to_save_raw_data}")

    if return_data:
        return communities_boundaries_json
    return None


@task(retries=3, retry_delay_seconds=2, name="Read escooter trips data")
def load_raw_data(
    rides_data_url: str = "https://data.cityofchicago.org/api/views/3rse-fbp6/rows.csv?accessType=DOWNLOAD&bom=true&format=true&delimiter=%3B",
    boundaries_url: str = "https://data.cityofchicago.org/api/geospatial/cauq-8yn6?method=export&format=GeoJSON",
    path_to_raw_data: str = "./data/raw",
):
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    rides_data_filepath = Path.joinpath(Path(path_to_raw_data), "rides_data.parquet")
    load_rides_data(rides_data_url, rides_data_filepath, return_data=True)

    boundaries_filepath = Path.joinpath(Path(path_to_raw_data), "boundaries.json")
    load_boundaries_data(boundaries_url, boundaries_filepath, return_data=True)


if __name__ == "__main__":
    load_raw_data()
