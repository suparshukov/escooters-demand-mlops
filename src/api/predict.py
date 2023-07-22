import json
import pickle
import sys
from datetime import date
from pathlib import Path

from flask import Flask, jsonify, request
from lightgbm import Booster
from shapely.geometry import Point, shape

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.features.build_features import get_center_lat_lon

MODEL = Booster(model_file='./models/model.txt')

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

with open("./references/community_codes_dict.pkl", 'rb') as community_codes_file:
    COMMUNITY_CODES_DICT = pickle.load(community_codes_file)

with open(Path.joinpath(Path('./data/raw'), "boundaries.json"), "r", encoding="utf-8") as boundaries_file:
    BOUNDARIES = json.load(boundaries_file)["features"]

LATITUDE, LONGITUDE = get_center_lat_lon(Path.joinpath(Path('./data/external/'), "city_center_coordinates.txt"))

COMMUNITY_GEOMETRY_DICT = {feature["properties"]["community"]: feature["geometry"] for feature in BOUNDARIES}
COMMUNITY_AREA_DICT = {
    community_name: shape(COMMUNITY_GEOMETRY_DICT[community_name]).area
    for community_name in COMMUNITY_GEOMETRY_DICT.keys()
}
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


app = Flask('trips-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    community_date = request.get_json()

    features = prepare_features(community_date)
    pred = predict(features)

    result = {'trips': pred}

    return jsonify(result)


# input_data = {"community": "LAKE VIEW", "date": "2020-09-20"}
# input_data = {"community": "ENGLEWOOD", "date": "2020-10-15"}
# input_data = {"community": "OHARE", "date": "2020-09-20"}


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
