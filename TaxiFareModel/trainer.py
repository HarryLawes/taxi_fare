# imports
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from TaxiFareModel.utils import compute_rmse, haversine_vectorized
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        #distance pipeline
        dist_pipe = Pipeline([('dist_transformer', DistanceTransformer()),
                              ('standardizer', StandardScaler())])
        #time pipeline
        time_pipe = Pipeline([('time_extractor', TimeFeaturesEncoder('pickup_datetime')),
                              ('encoder', OneHotEncoder())])
        #split distance and time cols
        dist_cols = ['pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude']
        time_cols = ['pickup_datetime']
        # create preprocessing pipeline
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, dist_cols),
                                    ('time', time_pipe, time_cols)])
        #model pipeline
        full_pipe = Pipeline([('preprocessing', preproc_pipe),
                            ('model', LinearRegression())])
        self.pipeline = full_pipe
        return self

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)
        return self

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)


if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    X = df.drop(columns=['fare_amount'])
    y = df['fare_amount']
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # train
    trainer = Trainer(X_train, y_train)
    trainer.run()
    # evaluate
    trainer.evaluate(X_test, y_test)
