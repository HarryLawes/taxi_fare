import pandas as pd
from google.cloud import storage

AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"

BUCKET_NAME = 'wagon-ml-lawes-00'
BUCKET_TRAIN_DATA_PATH = 'data/train_10k.csv'

def get_data(nrows=10_000):
    '''returns a DataFrame with nrows from s3 bucket'''
    client = storage.Client()
    df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}", nrows=nrows)
    return df


def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    if 'Unnamed: 0' in df.columns:
      df = df.drop('Unnamed: 0', 1)
    #df = df.set_index('key')
    return df


if __name__ == '__main__':
    df = get_data(nrows=1000)
