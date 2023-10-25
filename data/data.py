import numpy as np
import pandas as pd
# Scaling DatMinMaxScalera
from sklearn.preprocessing import MinMaxScaler

# Splitting data to train and test
# Cause we only have one data file
from sklearn.model_selection import train_test_split

import sys
sys.path.append("..")
from utils import scaler, rescaler


def read_excel_data(data: str) -> pd.DataFrame:
    """
    Read data from an Excel file.

    Args:
        data (str): Name of the Excel data file.

    Returns:
        pd.DataFrame: The data read from the Excel file.
    """
    df = pd.read_excel(data, sheet_name="Data", header=[1])
    return df

def process_data(df: pd.DataFrame, lags: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, callable, callable]:
    """
    Process Data
    Reshape and split data into train and test data.
    
    Parameters:
        df (df.DataFram): dataframe of the data
        lags (int): time lag
    
    Returns:
        X_train (np.ndarray)
        y_train (np.ndarray)
        X_test (np.ndarray)
        y_test (np.ndarray)
        flow_scaler (func)
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=.75)

    return X_train, y_train, X_test, y_test, flow_scaler, flow_rescaler, latlong_scaler

def process_data_prophet(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Process Data for Prophet Model
    Reshape and split data into train and test data for prophet model
    
    Parameters:
        df (df.DataFram): dataframe of the data
    
    Returns:
        train (np.ndarray)
        test (np.ndarray)
    """
    first_velo_col_pos = df.columns.get_loc("V00")
    flow_data = df.to_numpy()[:, first_velo_col_pos:]

    flatten_flow_data = flow_data.reshape(-1, 1)

    lat_data = df['NB_LATITUDE'].to_numpy().reshape(-1, 1)
    long_data = df['NB_LONGITUDE'].to_numpy().reshape(-1, 1)

    latlong_data = np.concatenate((lat_data, long_data), axis=1)

    date_data = df['Date'].to_numpy().reshape(-1, 1)

    num_time_steps = 96

    # 15 minutes per velo
    time_values = np.arange(-1, num_time_steps - 1) * 15 * 60 * 1000 * 1000 * 1000

    expanded_latlong = np.repeat(latlong_data, num_time_steps, axis=0).reshape(-1, 2)

    expanded_date = np.repeat(date_data, num_time_steps, axis=0).reshape(-1, num_time_steps)
    expanded_date = pd.to_datetime((expanded_date + time_values).reshape(-1, 1))

    train = np.hstack((expanded_latlong, expanded_date, flatten_flow_data))

    train_size = int(len(train) * 0.75)
    test = train[train_size:, :]
    train = train[:train_size, :]

    return train, test
