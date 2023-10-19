"""
Definition of NN model
"""
from keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten, BatchNormalization, LeakyReLU
from keras.layers import LSTM, GRU
from keras.models import Sequential, load_model
from keras.models import Model
from keras.regularizers import l1
from keras import Input, Model
from prophet import Prophet


def get_lstm(units):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(LSTM(units[2], return_sequences=True))
    model.add(LSTM(units[3]))
    model.add(Dropout(0.2))
    model.add(Dense(units[4], activation="sigmoid"))

    return model


def get_gru(units):
    """GRU(Gated Recurrent Unit)
    Build GRU Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(GRU(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(GRU(units[2], return_sequences=True))
    model.add(GRU(units[3]))
    model.add(Dropout(0.2))
    model.add(Dense(units[4], activation="sigmoid"))

    return model


def _get_sae(inputs, hidden, output):
    """SAE(Auto-Encoders)
    Build SAE Model.

    # Arguments
        inputs: Integer, number of input units.
        hidden: Integer, number of hidden units.
        output: Integer, number of output units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(Dense(hidden, input_dim=inputs, name='hidden'))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output, activation='sigmoid'))

    return model


def get_saes(layers):
    """SAEs(Stacked Auto-Encoders)
    Build SAEs Model.

    # Arguments
        layers: List(int), number of input, output and hidden units.
    # Returns
        models: List(Model), List of SAE and SAEs.
    """
    sae1 = _get_sae(layers[0], layers[1], layers[-1])
    sae2 = _get_sae(layers[1], layers[2], layers[-1])
    sae3 = _get_sae(layers[2], layers[3], layers[-1])

    saes = Sequential()
    saes.add(Dense(layers[1], input_dim=layers[0], name='hidden1'))
    # saes.add(BatchNormalization())
    # saes.add(LeakyReLU())
    saes.add(Activation('relu'))
    saes.add(Dense(layers[2], name='hidden2'))
    # saes.add(BatchNormalization())
    # saes.add(LeakyReLU())
    saes.add(Activation('relu'))
    saes.add(Dense(layers[3], name='hidden3'))
    # saes.add(BatchNormalization())
    # saes.add(LeakyReLU())
    saes.add(Activation('relu'))
    saes.add(Activation('relu'))
    saes.add(Dropout(0.2))
    saes.add(Dense(layers[4], activation='sigmoid'))

    models = [sae1, sae2, sae3, saes]

    return models

def get_cnn(units):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(Conv1D(units[1], kernel_size=3, input_shape=(units[0], 1), activation='relu'))
    model.add(Conv1D(units[2], kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model

def get_prophet():
  """
  This function creates and returns a Prophet model.

  Returns:
    Prophet: The Prophet model.
  """
  model = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
  model.add_seasonality(name='daily', period=1, fourier_order=15)

  return model
