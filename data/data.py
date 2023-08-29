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
        flow_scaler (MinMaxScaler)
        latlong_scaler (MinMaxScaler)
    """
    df = pd.read_excel(data, sheet_name="Data", header=[1])

    first_velo_col_pos = df.columns.get_loc("V00")
    flow_group = np.char.mod("V%02d", np.arange(0, 96))
    grouped = df.groupby(['NB_LATITUDE', 'NB_LONGITUDE'])[flow_group].apply(lambda x: x.values.tolist())

    flow_data = grouped.values
    flow_data = grouped.values
    flow_max = np.array(flow_data.max()).max()
    flow_min = np.array(flow_data.min()).min()
    def flow_scaler(x):
        return (x - flow_min) / (flow_max - flow_min)

    def flow_rescaler(x):
        return x * (flow_max - flow_min) + flow_min

    latlong_data = np.array(grouped.index.to_list())


    lags = 7
    train = []

    i = 0
    for flow in grouped.values:
        flow = np.array(flow, dtype=float).flatten()
        flow = np.vectorize(flow_scaler)(flow)
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

    return X_train, y_train, X_test, y_test, flow_scaler, flow_rescaler

