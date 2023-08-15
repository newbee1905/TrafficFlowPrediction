"""
Processing the data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def process_data(train, test, lags):
    """Process data
    Reshape and split train\test data.

    # Arguments
        train: String, name of .csv train file.
        test: String, name of .csv test file.
        lags: integer, time lag.
    # Returns
        X_train: ndarray.
        y_train: ndarray.
        X_test: ndarray.
        y_test: ndarray.
        scaler: StandardScaler.
    """
    df = pd.read_excel(train, sheet_name="Data", header=[1])

    first_velo_col_pos = df.columns.get_loc("V00")
    flow_data = df.to_numpy()[:, first_velo_col_pos:]

    flow_scaler = MinMaxScaler(feature_range=(0, 1)).fit(flow_data)
    flow_values = flow_scaler.transform(flow_data)

    lat_data = df['NB_LATITUDE'].to_numpy().reshape(-1, 1)
    long_data = df['NB_LONGITUDE'].to_numpy().reshape(-1, 1)

    lat_scaler = MinMaxScaler(feature_range=(0, 1)).fit(lat_data)
    long_scaler = MinMaxScaler(feature_range=(0, 1)).fit(long_data)

    lat = lat_scaler.transform(lat_data)
    long = long_scaler.transform(long_data)

    latlong = np.concatenate((lat, long), axis=1)

    train = []
    for i in range(0, len(flow_values)):
        for j in range(96):
            k = j
            if k == 95:
                k = -1
            # 15 minutes per velo
            time_of_vel = j * 15 / 24 / 60
            inputs = np.append([time_of_vel], latlong[i])
            inputs = np.append(inputs, flow_values[i, k + 1])
            train.append(inputs)
    train = np.array(train)

    np.random.shuffle(train)

    X_train = train[:, :-1]
    y_train = train[:, -1]

    return X_train, y_train, None, None, None
