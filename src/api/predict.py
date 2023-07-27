import json
import pickle
from datetime import date, datetime
from typing import Tuple

import mlflow.pyfunc
from flask import Flask, jsonify, request
from mlflow import MlflowClient
from shapely.geometry import Point, shape

TRACKING_URI = "http://16.171.140.74:5000"
# mlflow.set_tracking_uri(TRACKING_URI)

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


def load_model(model_name):
    stage = "Production"
    client = MlflowClient(registry_uri=TRACKING_URI)
    latest_version = client.get_latest_versions(model_name, stages=[stage])[0].source
    model = mlflow.pyfunc.load_model(model_uri=latest_version)
    print(f'{datetime.now()} Model loaded')

    return model


MODEL = load_model(MODEL_NAME)

with open("community_codes_dict.pkl", 'rb') as community_codes_file:
    COMMUNITY_CODES_DICT = pickle.load(community_codes_file)

with open("boundaries.json", "r", encoding="utf-8") as boundaries_file:
    BOUNDARIES = json.load(boundaries_file)["features"]

COMMUNITY_GEOMETRY_DICT = {feature["properties"]["community"]: feature["geometry"] for feature in BOUNDARIES}
COMMUNITY_AREA_DICT = {
    community_name: shape(COMMUNITY_GEOMETRY_DICT[community_name]).area
    for community_name in COMMUNITY_GEOMETRY_DICT.keys()
}


def get_center_lat_lon(path_to_file: str) -> Tuple[float, float]:
    """
    Function loads the coordinates of the city center from a file
    @param path_to_file:
    @return:
    """

    with open(path_to_file, "r", encoding="utf-8") as file_with_coordinates:
        city_center_coordinates = file_with_coordinates.read()
        lat, lon = [float(x) for x in city_center_coordinates.split(",")]

    return lat, lon


LATITUDE, LONGITUDE = get_center_lat_lon("city_center_coordinates.txt")

COMMUNITY_DISTANCE_DICT = {
    community_name: shape(COMMUNITY_GEOMETRY_DICT[community_name]).centroid.distance(Point(LONGITUDE, LATITUDE))
    for community_name in COMMUNITY_GEOMETRY_DICT.keys()
}


def prepare_features(input_data):
    calculated_features = dict()
    start_date = date.fromisoformat(input_data["date"])
    calculated_features['community'] = COMMUNITY_CODES_DICT[input_data["community"]]
    calculated_features['day_of_year'] = start_date.timetuple().tm_yday
    calculated_features['day_of_week'] = start_date.weekday()
    calculated_features['is_weekend'] = int(start_date.weekday() in {5, 6})
    calculated_features['week'] = start_date.isocalendar().week
    calculated_features['month'] = start_date.month
    calculated_features['area'] = COMMUNITY_AREA_DICT[input_data["community"]]
    calculated_features['distance_to_center'] = COMMUNITY_DISTANCE_DICT[input_data["community"]]

    features = [calculated_features[key] for key in MODEL_FEATURES]

    return features


def predict(features):
    pred = MODEL.predict([features])[0]
    return round(pred)


print(f'{datetime.now()} Starting app')
app = Flask('trips-prediction')
print(f'{datetime.now()} App started')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    community_date = request.get_json()
    print(f"{datetime.now()} Request: {community_date}")

    features = prepare_features(community_date)
    pred = predict(features)

    result = {'trips': pred, 'model_version': MODEL.metadata.run_id}

    return jsonify(result)


# input_data = {"community": "LAKE VIEW", "date": "2020-09-20"}
# input_data = {"community": "ENGLEWOOD", "date": "2020-10-15"}
# input_data = {"community": "OHARE", "date": "2020-09-20"}


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
