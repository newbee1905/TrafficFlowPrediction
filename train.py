"""
Train the NN model.
"""
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from data.data import process_data
from model import model
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
<<<<<<< Updated upstream
from multiprocessing import cpu_count, Process
=======
import sklearn.metrics as metrics
from multiprocessing import cpu_count, Pool
from prophet.serialize import model_to_json
>>>>>>> Stashed changes
warnings.filterwarnings("ignore")

from keras import backend as K


def train_model(model, X_train, y_train, name, config):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    K.set_value(model.optimizer.learning_rate, 0.01)
    early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=30, min_lr=0.001)
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        callbacks=[early, reduce_lr],
        validation_split=0.05,
        workers=cpu_count(),
        use_multiprocessing=True,
    )

    model.save('model/' + name + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/' + name + ' loss.csv', encoding='utf-8', index=False)


def train_seas(models, X_train, y_train, name, config):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    temp = X_train
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(input=p.input,
                                       output=p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])

        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)

        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer('hidden%d' % (i + 1)).set_weights(weights)

    train_model(saes, X_train, y_train, name, config)

<<<<<<< Updated upstream
=======
def train_prophet(model, data, name, config):
    """
    This function trains a Prophet model on the given data with regressors and saves the model and loss history.

    Args:
        model (Prophet): The Prophet model to train.
        data (DataFrame): The time series data to train the model on.
        regressor_data (DataFrame): The additional regressor data to include.
        name (str): The name of the model for saving.
        config (object): Configuration object with parameters like epochs.

    Returns:
        None
    """
    # model.add_regressor('lat')
    # model.add_regressor('lng')
    hist = model.fit(data)
    with open(f'model/{name}.json', 'w') as fout:
        fout.write(model_to_json(model))  # Save model
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(
        f'model/{name}_lost.csv',
        encoding='utf-8',
        index=False
    )

def train_cnn(model, X_train, y_train, name, config):
    model.compile(loss="mse", optimizer="adam", metrics=['mape'])
    K.set_value(model.optimizer.learning_rate, 0.01)
    early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=15, min_lr=0.001)
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        callbacks=[early, reduce_lr],
        validation_split=0.05,
        workers=cpu_count(),
        use_multiprocessing=True
    )

    model.save(f'{data_folder}/model/{name}.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(f'{data_folder}/model/{name}_loss.csv', encoding='utf-8', index=False)

>>>>>>> Stashed changes

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="lstm",
        help="Model to train.")
    parser.add_argument(
        "--all",
        action='store_true',
        default=False,
        help="Train all models.")
<<<<<<< Updated upstream
=======
    parser.add_argument(
        "--lags",
        type=int,
        default=7,
        help="Lags in the model.")
    parser.add_argument(
        "--batch",
        type=int,
        default=8192,
        help="Batch for model training.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1800,
        help="Epochs for model training.")
>>>>>>> Stashed changes
    args = parser.parse_args()

    lag = 7
    config = {"batch": 8192, "epochs": 600}
    file = 'data/Scats Data October 2006.xls'
    X_train, y_train, _, _, _ = process_data(file, lag)

    if args.all == True:
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            lstm_proc = Process(target=train_model, args=(model.get_lstm([9, 64, 64, 1]), X_train, y_train, "lstm", config,))
            gru_proc = Process(target=train_model, args=(model.get_gru([9, 64, 64, 1]), X_train, y_train, "gru", config,))
            lstm_proc.start()
            gru_proc.start()
            procs = [lstm_proc, gru_proc]
            for proc in procs:
                proc.join()
    else:
        if args.model == 'lstm':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = model.get_lstm([9, 64, 64, 1])
            train_model(m, X_train, y_train, args.model, config)
        if args.model == 'gru':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = model.get_gru([9, 64, 64, 1])
            train_model(m, X_train, y_train, args.model, config)
        if args.model == 'saes':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
            m = model.get_saes([9, 400, 400, 400, 1])
            train_seas(m, X_train, y_train, args.model, config)


if __name__ == '__main__':
    main(sys.argv)
