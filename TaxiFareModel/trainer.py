# imports
import mlflow
from  mlflow.tracking import MlflowClient
from memoized_property import memoized_property
from google.cloud import storage

import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from TaxiFareModel.utils import compute_rmse, haversine_vectorized, minkowski_distance_gps
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder, DistanceFromCentre, \
                                   ManhattanDistance
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

MLFLOW_URI = "https://mlflow.lewagon.co/"
myname = "H_LAWES"
EXPERIMENT_NAME = f"[UK] [LONDON] [524_{myname}] TaxiFareModel 0.0"

BUCKET_NAME = 'wagon-ml-lawes-00'
STORAGE_LOCATION = 'models/taxifare/model.joblib'

def upload_model_to_gcp():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STORAGE_LOCATION)
    blob.upload_from_filename('model.joblib')

class Trainer():
    def __init__(self, X, y, iteration=0):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.iteration = iteration

    def set_pipeline(self, estimator):
        """defines the pipeline as a class attribute"""
        #distance pipeline
        dist_pipe = Pipeline([('dist_transformer', DistanceTransformer()), 
                              ('standardizer', StandardScaler())])
        #time pipeline
        time_pipe = Pipeline([('time_extractor', TimeFeaturesEncoder('pickup_datetime')),
                              ('encoder', OneHotEncoder())])
        #split distance and time cols
        dist_cols = ['pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude']
        #dfc_cols = ['pickup_longitude', 'pickup_latitude']
        time_cols = ['pickup_datetime']
        # create preprocessing pipeline
        # if self.iteration > 0:
        #     man_dist_pipe = Pipeline([('dist_centre', ManhattanDistance()),  
        #                               ('standardizer', StandardScaler())])
        #     preproc_pipe = ColumnTransformer([('manhattan', man_dist_pipe, dist_cols),
        #                                       ('time', time_pipe, time_cols)])
        # else:
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, dist_cols),
                                              ('time', time_pipe, time_cols)])
        #model pipeline
        full_pipe = Pipeline([('preprocessing', preproc_pipe),
                              ('model', estimator)])
        self.pipeline = full_pipe
        return self

    def run(self, estimator):
        """set and train the pipeline"""
        self.set_pipeline(estimator)
        self.pipeline.fit(self.X, self.y)
        return self

    def evaluate(self, X_test, y_test, estimator):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        # self.mlflow_log_metric('rmse', rmse)
        # self.mlflow_log_param('model', f'{estimator}')
        # if self.iteration > 0:
        #     feature = 'with'
        # else:
        #     feature = 'without'
        # self.mlflow_log_param('manhattan_distance', f'{feature}')

    def save_model(self, estimator):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, 'model.joblib')
        upload_model_to_gcp()

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(EXPERIMENT_NAME)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

knn = KNeighborsRegressor()
rf = RandomForestRegressor()
svr = SVR()
lasso = Lasso()
ridge = Ridge()

models = [rf]

if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    X = df.drop(columns=['fare_amount'])
    y = df['fare_amount']
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    for estimator in models:
        # train
        trainer = Trainer(X_train, y_train)
        trainer.run(estimator)
        # evaluate
        #trainer.evaluate(X_test, y_test, estimator)
        # save
        trainer.save_model(estimator)