import pandas as pd

class MissingValueStrategy:
    def fill(self, df: pd.DataFrame):
        raise NotImplementedError


class MeanStrategy(MissingValueStrategy):
    def fill(self, df):
        return df.fillna(df.mean(numeric_only=True))


class MedianStrategy(MissingValueStrategy):
    def fill(self, df):
        return df.fillna(df.median(numeric_only=True))


class MinStrategy(MissingValueStrategy):
    def fill(self, df):
        return df.fillna(df.min(numeric_only=True))


class MaxStrategy(MissingValueStrategy):
    def fill(self, df):
        return df.fillna(df.max(numeric_only=True))


class DropStrategy(MissingValueStrategy):
    def fill(self, df):
        return df.dropna()
