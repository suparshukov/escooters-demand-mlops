"""Module providing functions for scraping external data"""
import logging
from pathlib import Path
from typing import Tuple

import click
import requests
from bs4 import BeautifulSoup


def scrape_city_center_coordinates(
    url: str, city_center_filepath: Path
) -> Tuple[float, float]:
    """
    Scrapes data on city center coordinates and saves it to a text file
    @param url: str, the address of the page to scrape
    the coordinates of the city center from
    @param city_center_filepath: Path, the path of a file
    to save the city center coordinates into
    @return: Tuple[float, float], the coordinates of the city center
    """

    if city_center_filepath.exists():
        with open(city_center_filepath, "r", encoding="utf-8") as file_with_coords:
            coordinates_str_list = file_with_coords.read().split(",")
            lat, lon = float(coordinates_str_list[0]), float(coordinates_str_list[1])
            print("skipping scrapping")
    else:
        html_doc = requests.get(url, allow_redirects=True, timeout=3).content
        soup = BeautifulSoup(html_doc, "html.parser")
        coordinates_element = soup.find_all(
            lambda tag: len(tag.find_all("p")) == 0
            and "Latitude and longitude coordinates are:" in tag.text
        )[0]
        coordinates_str = coordinates_element.find("strong").text
        city_center_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(
            city_center_filepath, "w", encoding="utf-8"
        ) as file_with_coords:
            file_with_coords.write(coordinates_str)
        print(coordinates_str)
        coordinates_str_list = coordinates_str.split(",")
        lat, lon = float(coordinates_str_list[0]), float(coordinates_str_list[1])
        print("scrapping finished")

    return lat, lon


@click.command()
@click.option("--coordinates_page_url", default="https://www.latlong.net/place/chicago-il-usa-1855.html", help="Url of a page with the city center coordinates")
@click.option("--path_to_external_data", default="./data/external", help="Path to external data")
def main(coordinates_page_url: str, path_to_external_data: str):
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    city_center_coordinates_filepath = Path.joinpath(
        Path(path_to_external_data), "city_center_coordinates.txt"
    )
    scrape_city_center_coordinates(
        coordinates_page_url, city_center_coordinates_filepath
    )


if __name__ == "__main__":
    main()
