# -*- coding: utf-8 -*-
from pathlib import Path

import pytest

from src.data.load_data import load_boundaries_data, load_rides_data


@pytest.fixture
def rides_dataset():
    """
    A fixture for rhe rides data
    @return: pandas DataFrame, the rides dataset
    """

    project_dir = Path(__file__).resolve().parents[1]
    url = (
        "https://data.cityofchicago.org/api/views/3rse-fbp6/rows.csv"
        "?accessType=DOWNLOAD&bom=true&format=true&delimiter=%3B"
    )
    raw_filepath = Path.joinpath(project_dir, "data/raw/rides_data.parquet")
    data = load_rides_data(url, raw_filepath, return_data=True)

    return data


@pytest.fixture
def booundaries():
    """
    A fixture for rhe districts boundaries data
    @return: pandas DataFrame, the districts boundaries data
    """

    project_dir = Path(__file__).resolve().parents[1]
    boundaries_url = "https://data.cityofchicago.org/api/geospatial/" "cauq-8yn6?method=export&format=GeoJSON"
    boundaries_filepath = Path.joinpath(project_dir, "data/raw/boundaries.json")
    booundaries = load_boundaries_data(boundaries_url, boundaries_filepath, return_data=True)

    return booundaries


def test_rides_dataset_length(rides_dataset):
    """
    Function for testing the rides dataset size
    """
    assert len(rides_dataset) == 630816


def test_rides_dataset_columns(rides_dataset):
    """
    Function for testing the rides dataset columns
    """
    assert list(rides_dataset.columns) == [
        "Trip ID",
        "Start Time",
        "End Time",
        "Trip Distance",
        "Trip Duration",
        "Vendor",
        "Start Community Area Number",
        "End Community Area Number",
        "Start Community Area Name",
        "End Community Area Name",
        "Start Centroid Latitude",
        "Start Centroid Longitude",
        "Start Centroid Location",
        "End Centroid Latitude",
        "End Centroid Longitude",
        "End Centroid Location",
    ]


def test_areas(rides_dataset, booundaries):
    """
    Function for testing the match of districts in the rides
    dataset and in the districts boundaries data
    """
    rides_data_areas = sorted(
        list([area_name for area_name in rides_dataset["Start Community Area Name"].unique() if area_name is not None])
    )

    boundaries_community_names = sorted(
        list([feature["properties"]["community"] for feature in booundaries["features"]])
    )

    assert len(rides_data_areas) == 77
    assert rides_data_areas == boundaries_community_names
