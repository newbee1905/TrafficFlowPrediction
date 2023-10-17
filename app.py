from fastapi import FastAPI
from enum import Enum 
from pydantic import BaseModel
import pandas as pd
import numpy as np
from keras.models import load_model
from data.data import read_excel_data, process_data
import time

from utils import scaler, rescaler

import uvicorn

app = FastAPI()

file = 'data/Scats Data October 2006.xls'
df = read_excel_data(file)
_, _, _, _, flow_scaler, flow_rescaler, latlong_scaler = process_data(df, 7)

first_velo_col_pos = df.columns.get_loc("V00")
flow_group = np.char.mod("V%02d", np.arange(0, 96))
grouped = df.groupby(['NB_LATITUDE', 'NB_LONGITUDE'])[flow_group].apply(lambda x: x.values.tolist())

lats = grouped.index.get_level_values('NB_LATITUDE').values
lngs = grouped.index.get_level_values('NB_LONGITUDE').values


def find_closest_value(lat, lng):
    # calculating the distance of all lat lng compare to the
    # given one
    distances = np.sqrt((lats - lat) ** 2 + (lngs - lng) ** 2)

    # get index of the one closest to 0
    closest_index = np.argmin(distances)

    return grouped.index[closest_index]

def gen_fake_flow(flows, start_time):
    hour, minute = start_time.split(':')
    time_pos = int(hour) * 4 + int(int(minute) / 15) + 7
    diff = np.floor(np.random.rand(7) * 10 - 5).astype(int)
    flows = np.append(flows[-7:], flows)
    return flows[time_pos - 7:time_pos] + diff

class ModelEnum(str, Enum):
    lstm = "lstm"
    gru = "gru"
    saes = "saes"

class PredTrafficFlow(BaseModel):
    start_time: str
    lat: float
    lng: float
    model: ModelEnum


models = {
    "lstm": load_model('model/lstm.h5'),
    "gru": load_model('model/gru.h5'),
    "saes": load_model('model/saes.h5'),
}

@app.post("/")
def root(pred: PredTrafficFlow) -> float:
    start_time = time.time()
    lat, lng = find_closest_value(pred.lat, pred.lng)
    latlong = np.array([lat, lng])
    latlong = latlong_scaler.transform(latlong.reshape(-1, 1)).reshape(-1, 2)
    print("--- Getting Lat Long ---\n--- %s seconds ---\n" % (time.time() - start_time))

    start_time = time.time()
    flows = gen_fake_flow(grouped[(lat, lng)], pred.start_time)
    print("--- Generating Fake Flows ---\n--- %s seconds ---\n" % (time.time() - start_time))

    start_time = time.time()
    X = np.array([np.append(latlong, np.vectorize(flow_scaler)(flows))])

    if pred.model == 'saes':
        X = np.reshape(X, (X.shape[0], X.shape[1]))
    else:
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    print("--- Reshape Flow ---\n--- %s seconds ---\n" % (time.time() - start_time))

    start_time = time.time()
    predict = models[pred.model].predict(X)
    predict = np.vectorize(flow_rescaler)(predict)
    print("--- Predicting ---\n--- %s seconds ---\n" % (time.time() - start_time))

    return predict[0][0]
