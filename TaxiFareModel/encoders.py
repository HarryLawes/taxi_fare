from sklearn.base import BaseEstimator, TransformerMixin
from TaxiFareModel.utils import haversine_vectorized, minkowski_distance, minkowski_distance_gps, \
                                deg2rad, rad2dist, lng_dist_corrected
import pandas as pd

class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """Extract the day of week (dow), the hour, the month and the year from a time column."""

    def __init__(self, time_column, time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'"""
        X.index = pd.to_datetime(X[self.time_column])
        X.index = X.index.tz_convert(self.time_zone_name)
        X["dow"] = X.index.weekday
        X["hour"] = X.index.hour
        X["month"] = X.index.month
        X["year"] = X.index.year
        return X[['dow', 'hour', 'month', 'year']]

class DistanceTransformer(BaseEstimator, TransformerMixin):
    """Compute the haversine distance between two GPS points."""

    def __init__(self, 
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude", 
                 end_lat="dropoff_latitude", 
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only one column: 'distance'"""
        X['distance'] = haversine_vectorized(X)
        return X[['distance']]

class DistanceFromCentre(BaseEstimator, TransformerMixin):
    """Compute the haversine distance from NYC Centre"""

    def __init__(self, 
                 start_lat="nyc_latitude",
                 start_lon="nyc_longitude", 
                 end_lat="pickup_latitude", 
                 end_lon="pickup_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only one column: 'distance_to_center'"""
        nyc_center = (40.7141667, -74.0063889)
        X["nyc_latitude"], X["nyc_longitude"] = nyc_center[0], nyc_center[1]
        args =  dict(start_lat="nyc_latitude", start_lon="nyc_longitude",
                    end_lat="pickup_latitude", end_lon="pickup_longitude")
        X['distance_to_center'] = haversine_vectorized(X, **args)
        return X[['distance_to_center']]

class ManhattanDistance(BaseEstimator, TransformerMixin):
    """Compute the haversine distance from NYC Centre"""

    def __init__(self, 
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude", 
                 end_lat="dropoff_latitude", 
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only one column: 'manhattan_dist'"""
        X['manhattan_dist'] = minkowski_distance_gps(X['pickup_longitude'], X['dropoff_longitude'],
                                       X['pickup_latitude'], X['dropoff_latitude'], 1)
        return X[['manhattan_dist']]