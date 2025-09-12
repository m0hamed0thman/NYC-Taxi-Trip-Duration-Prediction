import numpy as np
import pandas as pd
from .strategies import MissingValueStrategy


class DataPreprocessor:
    def __init__(self, missing_strategy: MissingValueStrategy, min_mins, max_mins, min_distance, max_distance, number_of_passengers):
        self.missing_strategy = missing_strategy
        self.min_mins = min_mins
        self.max_mins = max_mins
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.number_of_passengers = number_of_passengers

    def drop_columns(self, df, cols):
        return df.drop(cols, axis=1)

    def nyc_boundaries(self, df):
        min_long, max_long = -74.3, -73.6
        min_lat, max_lat = 40.4, 41.0
        df = df[(df['pickup_longitude'].between(min_long, max_long)) &
                (df['pickup_latitude'].between(min_lat, max_lat))]
        df = df[(df['dropoff_longitude'].between(min_long, max_long)) &
                (df['dropoff_latitude'].between(min_lat, max_lat))]
        return df

    def duration_boundaries(self, df, min_mins=1, max_mins=180):
        return df[(df['trip_duration'] > min_mins*60) & (df['trip_duration'] < max_mins*60)]


    def passengers_boundaries(self, df, number_of_trips = 4):
        return df[(df['passenger_count'] <= number_of_trips)]


    def haversine_array(self, lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    def extract_distance(self, df):
        df["trip_distance"] = self.haversine_array(
            df["pickup_latitude"], df["pickup_longitude"],
            df["dropoff_latitude"], df["dropoff_longitude"])
        return df

    def distance_boundaries(self, df, min_distance, max_distance):
        return df[(df["trip_distance"] >= min_distance) & (df["trip_distance"] <= max_distance)]

    def extract_time(self, df):
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
        df["month"] = df["pickup_datetime"].dt.month
        df["day"] = df["pickup_datetime"].dt.day
        df["hour"] = df["pickup_datetime"].dt.hour
        df["minute"] = df["pickup_datetime"].dt.minute
        df["weekday_num"] = df["pickup_datetime"].dt.dayofweek
        return df

    def fill_missing(self, df):
        return self.missing_strategy.fill(df)

    def process(self, df):
        df = self.drop_columns(df, ["id", "store_and_fwd_flag"])
        df = self.nyc_boundaries(df)
        df = self.extract_distance(df)
        df = self.distance_boundaries(df,self.min_distance,self.max_distance)
        df = self.drop_columns(df, ["pickup_longitude", "pickup_latitude",
                                    "dropoff_longitude", "dropoff_latitude"])
        df = self.extract_time(df)
        df = self.drop_columns(df, ["pickup_datetime", "minute"])
        df = self.duration_boundaries(df, self.min_mins,self.max_mins)
        df = self.fill_missing(df)
        df = self.passengers_boundaries(df,self.number_of_passengers)
        df = df.drop_duplicates()
        return df
