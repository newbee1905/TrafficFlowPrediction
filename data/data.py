#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries
# 
# We will use numpy, pandas and sklearn scales and data splitter

# In[ ]:


import numpy as np
import pandas as pd
# Scaling Data
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Splitting data to train and test
# Cause we only have one data file
from sklearn.model_selection import train_test_split


# # Functions

# In[ ]:


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
        scaler (StandardScaler)
    """
    df = pd.read_excel(data, sheet_name="Data", header=[1])

    first_velo_col_pos = df.columns.get_loc("V00")
    flow_data = df.to_numpy()[:, first_velo_col_pos:]

    flow_scaler = MinMaxScaler(feature_range=(0, 1)).fit(flow_data)
    flow_values = flow_scaler.transform(flow_data)

    lat_data = df['NB_LATITUDE'].to_numpy().reshape(-1, 1)
    long_data = df['NB_LONGITUDE'].to_numpy().reshape(-1, 1)

    latlong_data = np.concatenate((lat_data, long_data), axis=1)

    latlong_scaler = MinMaxScaler(feature_range=(0, 1)).fit(latlong_data)

    latlong = latlong_scaler.transform(latlong_data)

    num_time_steps = 96
    #
    # 15 minutes per velo
    time_values = np.arange(num_time_steps) * 15 / 24 / 60
    time_column = np.tile(time_values, len(flow_values)).reshape(-1, 1)

    expanded_latlong = np.repeat(latlong, num_time_steps, axis=0).reshape(-1, 2)

    shifted_flow_values = np.roll(flow_values, -1, axis=1)
    shifted_flow_column = shifted_flow_values.reshape(-1, 1)
    
    train = np.hstack((time_column, expanded_latlong, shifted_flow_column))

    np.random.shuffle(train)

    X = train[:, :-1]
    y = train[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size = .75)

    return X_train, y_train, X_test, y_test, flow_scaler 

