import sys
from pathlib import Path

from prefect import flow

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.load_data import load_raw_data
from src.data.scrape_data import scrape_external_data
from src.data.split import split_dataset
from src.features.build_features import featurize
from src.models.hpo import hpo
from src.models.train_model import train_log_model


@flow(name="Model training")
def main_flow():
    load_raw_data()
    scrape_external_data()
    featurize()
    split_dataset()
    hpo()
    train_log_model()


if __name__ == "__main__":
    main_flow()
