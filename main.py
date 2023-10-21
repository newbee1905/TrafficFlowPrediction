"""
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).
"""
import argparse
import math
import warnings
import numpy as np
import pandas as pd
from data.data import read_excel_data, process_data, process_data_prophet
from keras.models import load_model
from prophet.serialize import model_from_json
import json
# from keras.utils.vis_utils import plot_model
# from tensorflow.keras.utils import plot_model
from keras.utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
warnings.filterwarnings("ignore")


def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """

    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape


def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """

    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)

def plot_results(y_true, y_preds, names, periods, cols=3):
    """Plot multiple subplots in a grid, each comparing true data to one prediction method.
    
    # Arguments
        y_true: List/ndarray, true data.
        y_preds: List/ndarray, list of predicted data from various methods.
        names: List, method names.
        periods: int, number of time periods to display in plots.
        cols: int, number of columns of figures
    """
    d = '2006-10-01 00:00'
    y_true = y_true[:periods]
    x = pd.date_range(d, periods=periods, freq='15min')

    # Calculate rows and columns for subplots grid
    total = len(y_preds) + 1
    rows = total // cols 
    rows += total % cols
    position = range(1, total + 1)

    # Generating unique colours
    colours = plt.cm.tab10(np.linspace(0, 1, len(y_preds) + 1))

    fig = plt.figure(figsize=(15, 5 * rows))

    # Plotting the true data
    ax = fig.add_subplot(rows, cols, position[0])
    ax.plot(x, y_true, label='True Data', color=colours[0])
    ax.set_title('True Data')
    ax.grid(True)
    ax.set_ylabel('Flow')

    date_format = mpl.dates.DateFormatter("%H:%M")

    # Plotting each prediction
    for i, (name, y_pred, color) in enumerate(zip(names, y_preds, colours[1:])):
        ax = fig.add_subplot(rows, cols, position[i + 1])
        ax.plot(x, y_true, label='True Data', color=colours[0])
        ax.plot(x, y_pred, label=name, color=color)
        ax.set_title(name)
        ax.grid(True)
        ax.set_ylabel('Flow')
        ax.legend()

        ax.xaxis.set_major_formatter(date_format)

    fig.autofmt_xdate()

    # Save the figure
    now = datetime.now()
    current_time = now.strftime("%m-%d-%Y-%H-%M-%S")
    plt.savefig(f'output-{current_time}.png', dpi=2400)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--days",
        default=1,
        help="Number of days in the plots.")
    parser.add_argument(
        "--hours",
        action='store_true',
        default=24,
        help="Number of hours per day in the plots.")
    parser.add_argument(
        "--lags",
        type=int,
        default=7,
        help="Lags in the model.\nPlease use the same one you use for train.py")
    args = parser.parse_args()

    lstm = load_model('model/lstm.h5')
    gru = load_model('model/gru.h5')
    saes = load_model('model/saes.h5')
    cnn = load_model('model/cnn.h5')
    with open('model/prophet.json', 'rb') as fin:
        prophet = model_from_json(fin.read())
    models = [lstm, gru, saes, cnn, prophet]
    names = ['LSTM', 'GRU', 'SAEs', 'CNN', 'Prophet']

    file = 'data/Scats Data October 2006.xls'
    df = read_excel_data(file)
    _, _, X_test, y_test, _, flow_rescaler, _ = process_data(df, args.lags)
    y_test = np.vectorize(flow_rescaler)(y_test)
    _, prophet_test = process_data_prophet(df)

    periods = args.days * args.hours * 60 // 15

    y_preds = []
    for name, model in zip(names, models):
        if name == 'Prophet':
            continue
        if name == 'SAEs':
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        else:
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        file = 'images/' + name + '.png'
        plot_model(model, to_file=file, show_shapes=True)
        predicted = model.predict(X_test)
        predicted = np.vectorize(flow_rescaler)(predicted)
        print(name)
        eva_regress(y_test, predicted)
        y_preds.append(predicted[:periods])
    print("Prophet")
    predicted = prophet.predict(pd.DataFrame(prophet_test, columns=["lat", "lng", "ds", "y"]))
    eva_regress(prophet_test[:, 3], predicted["yhat"])
    y_preds.append(predicted["yhat"][:periods])

    plot_results(y_test[:periods], y_preds, names, periods)


if __name__ == '__main__':
    main()
