import json
import os

import pandas as pd
from lightgbm import LGBMRegressor


def train_model(train: pd.DataFrame, model_features, categorical_features, model_params):
    (X_train, y_train) = (train[model_features], train["rides_number"])
    X_train[categorical_features] = X_train[categorical_features].astype("category")

    lgbm = LGBMRegressor(objective="poisson", **model_params)

    lgbm.fit(X_train, y_train)

    return lgbm


def main():

    train = pd.read_parquet(os.path.join("./data/processed", "train.parquet"))
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

    model_params_path = "./models/best_params.json"
    with open(model_params_path, "r") as f:
        model_params = json.load(f)

    categorical_features = ['community', 'day_of_week', 'is_weekend']
    model = train_model(train, model_features, categorical_features, model_params)
    model.booster_.save_model("./models/model.txt")


if __name__ == "__main__":
    main()


# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#
# import seaborn as sns
# test = pd.read_parquet('../data/processed/test.parquet')
# X_test, y_test = test[model_features], test['rides_number']
# X_test[categorical_features] = X_test[categorical_features].astype("category")
# X_test["week"] = X_test["week"].astype(int)
#
# y_pred = model.predict(X_test)
#
# X_test['y_true'] = y_test
# X_test['y_pred'] = np.round(y_pred)
#
# mean_absolute_error(X_test['y_true'], X_test['y_pred'])
# mean_squared_error(X_test['y_true'], X_test['y_pred'], squared=False)
# r2_score(X_test['y_true'], X_test['y_pred'])
#
# sns.histplot(X_test[['y_pred', 'y_true']])
#
# sns.lineplot(X_test.groupby('day_of_year').agg({'y_true': 'sum', 'y_pred': 'sum'}))
#
# pass
