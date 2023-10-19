#!/usr/bin/env python
# coding: utf-8

# Importing Libraries
# We will use numpy, pandas and sklearn scales and data splitter
import numpy as np
import pandas as pd
<<<<<<< Updated upstream
# Scaling Data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
=======
# Scaling DatMinMaxScalera
from sklearn.preprocessing import MinMaxScaler, StandardScaler
>>>>>>> Stashed changes

# Splitting data to train and test
# Cause we only have one data file
from sklearn.model_selection import train_test_split

<<<<<<< Updated upstream

# # Functions

# In[ ]:

=======
import sys
sys.path.append("..")
from utils import scaler, rescaler
from typing import Callable

# Functions
def read_excel_data(data: str) -> pd.DataFrame:
    """
    Read data from an Excel file.
>>>>>>> Stashed changes

def process_data(data: str, lags: int) -> (np.ndarray, np.ndarray, np.ndarray, StandardScaler):
    """
    Process Data
    Reshape and split data into train and test data.
    
    Parameters:
        data (str): name of the excel data file
        lags (int): time lag
    
    Returns:
        X_train (np.ndarray)
        y_train (np.ndarray)
        X_test (np.ndarray)
        y_test (np.ndarray)
        flow_scaler (MinMaxScaler)
        latlong_scaler (MinMaxScaler)
    """
    df = pd.read_excel(data, sheet_name="Data", header=[1])

    # first_velo_col_pos = df.columns.get_loc("V00")
    # flow_data = df.to_numpy()[:, first_velo_col_pos:]

    # # flow_scaler = MinMaxScaler(feature_range=(0, 1)).fit(flow_data)
    # # flow_values = flow_scaler.transform(flow_data)

    # flatten_flow_data = flow_data.reshape(-1, 1)
    # flow_scaler = MinMaxScaler(feature_range=(0, 1)).fit(flatten_flow_data)
    # flow_values = flow_scaler.transform(flatten_flow_data)

    # lat_data = df['NB_LATITUDE'].to_numpy().reshape(-1, 1)
    # long_data = df['NB_LONGITUDE'].to_numpy().reshape(-1, 1)

    # latlong_data = np.concatenate((lat_data, long_data), axis=1)

    # latlong_scaler = MinMaxScaler(feature_range=(0, 1)).fit(latlong_data)

    # latlong = latlong_scaler.transform(latlong_data)

    # num_time_steps = 96
    
    # # 15 minutes per velo
    # time_values = np.arange(num_time_steps) * 15 / 24 / 60
    # time_column = np.tile(time_values, len(flow_data)).reshape(-1, 1)

    # # time_values = np.array(np.repeat(df['Date'], 96)).reshape(96, -1)
    # # time_values_diff = np.tile(pd.to_timedelta(np.char.mod('%dmin', (np.arange(96) - 1) * 15)), len(df['Date'])).reshape(96, -1)

    # # time_values += time_values_diff

    # # time_values = time_values.reshape(-1, 1)

    # # time_scaler = MinMaxScaler(feature_range=(0, 1)).fit(time_values)
    # # time_column = time_scaler.transform(time_values)

    # expanded_latlong = np.repeat(latlong, num_time_steps, axis=0).reshape(-1, 2)

    # # shifted_flow_values = np.roll(flow_values, -1, axis=1)
    # # shifted_flow_column = shifted_flow_values.reshape(-1, 1)

    # shifted_flow_column = np.roll(flow_values, -1)

    
    # train = np.hstack((time_column, expanded_latlong, shifted_flow_column))

    # np.random.shuffle(train)
    first_velo_col_pos = df.columns.get_loc("V00")
    flow_group = np.char.mod("V%02d", np.arange(0, 96))
    grouped = df.groupby(['NB_LATITUDE', 'NB_LONGITUDE'])[flow_group].apply(lambda x: x.values.tolist())

    flow_data = grouped.values
    flow_scaler = np.array(flow_data.max()).max()

    latlong_data = np.array(grouped.index.to_list())


    lags = 7
    train = []

    i = 0
    for flow in grouped.values:
        flow = np.array(flow, dtype=float).flatten() / flow_scaler
        indices = np.arange(lags, len(flow))
        offset = np.arange(-lags, 1)
        flow = flow[indices[:, np.newaxis] + offset]
        latlong = np.tile(latlong_data[i], (len(flow), 1))
        combined_arr = np.hstack((latlong, flow))
        train.extend(combined_arr)
        i += 1

    train = np.array(train)

    X = train[:, :-1]
    y = train[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size = .75)

    return X_train, y_train, X_test, y_test, flow_scaler

def process_data_cnn(df: pd.DataFrame, lags: int, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Process Data
    Reshape and split data into train and test data.

    Parameters:
        df (df.DataFram): dataframe of the data
        lags (int): time lag
        window_size (int): Size of the CNN window

    Returns:
        X_train (np.ndarray)
        y_train (np.ndarray)
        X_test (np.ndarray)
        y_test (np.ndarray)
    """

    flow_group = np.char.mod("V%02d", np.arange(0, 96))
    grouped = df.groupby(['NB_LATITUDE', 'NB_LONGITUDE'])[flow_group].apply(lambda x: x.values.tolist())

    flow_data = grouped.values
    flow_max = np.array(flow_data.max()).max()
    flow_min = np.array(flow_data.min()).min()

    flow_scaler = scaler(flow_min, flow_max)
    flow_rescaler = rescaler(flow_min, flow_max)

    latlong_data = np.array(grouped.index.to_list())
    latlong_scaler = MinMaxScaler(feature_range=(0, 1)).fit(latlong_data.reshape(-1, 1))
    latlong_data = latlong_scaler.transform(latlong_data.reshape(-1, 1)).reshape(-1, 2)

    train = []
    i = 0
    for flow in grouped.values:
        flow = np.array(flow, dtype=float).flatten()
        flow = np.vectorize(flow_scaler)(flow)
        indices = np.arange(lags, len(flow) - window_size + 1)
        offset = np.arange(-lags, 1)
        flow = flow[indices[:, np.newaxis] + offset]
        latlong = np.tile(latlong_data[i], (len(flow), 1))
        combined_arr = np.hstack((latlong, flow))
        train.extend(combined_arr)
        i += 1

    train = np.array(train)
    X = train[:, :-1]
    y = train[:, -1]

    # Reshape X to be suitable for 1D CNN
    X = X.reshape(-1, window_size, 7)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=.75)

    return X_train, y_train, X_test, y_test

