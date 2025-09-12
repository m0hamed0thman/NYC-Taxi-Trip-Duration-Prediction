import argparse

import pandas as pd
from sklearn.preprocessing import StandardScaler
from preprocessing.preprocessor import *
from preprocessing.strategies import *
from models.factory import ModelFactory
from utils.metrics import r2, rmse

def parse_args():
    parser = argparse.ArgumentParser(description="Train and validate Taxi Trip Duration model")

    # Preprocessing
    parser.add_argument("--missing_strategy", type=str, choices=["mean", "median", "min", "max", "drop"],
                        default="median", help="Strategy for handling missing values")
    parser.add_argument("--min_mins", type=int, default=1)
    parser.add_argument("--max_mins", type=int, default=180)
    parser.add_argument("--min_distance", type=float, default=0.0)
    parser.add_argument("--max_distance", type=float, default=100.0)
    parser.add_argument("--number_of_passengers", type=int, default=4)

    # Model
    parser.add_argument("--model", type=str, default="ridge",
                        choices=["ridge", "lasso", "linear", "polynomial", "poly_ridge", "poly_lasso"],
                        help="Type of regression model")

    return parser.parse_args()


def get_missing_strategy(name: str):
    strategies = {
        "mean": MeanStrategy(),
        "median": MedianStrategy(),
        "min": MinStrategy(),
        "max": MaxStrategy(),
        "drop": DropStrategy()
    }
    return strategies[name]



if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    args = parse_args()
    # Load training data
    df = pd.read_csv("/home/mohamed-othman/PycharmProjects/Taxi Trip Duration Prediction/data/train.csv")

    # Preprocess
    preprocessor = DataPreprocessor(
        missing_strategy=get_missing_strategy(args.missing_strategy),
        min_mins=args.min_mins,
        max_mins=args.max_mins,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        number_of_passengers=args.number_of_passengers
    )
    df = preprocessor.process(df)

    y_train = df["trip_duration"]
    X_train = df.drop(["trip_duration"], axis=1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Train model
    model = ModelFactory.create_model("ridge")
    model.fit(X_train, y_train)

    # Validation
    vdf = pd.read_csv("/home/mohamed-othman/PycharmProjects/Taxi Trip Duration Prediction/data/val.csv")
    vdf = preprocessor.process(vdf)
    y_val = vdf["trip_duration"]
    X_val = scaler.transform(vdf.drop(["trip_duration"], axis=1))

    y_pred = model.predict(X_val)

    print("RMSE:", rmse(y_val, y_pred))
    print("R²:", r2(y_val, y_pred))
