import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input # ANN
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN # RNN
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D # CNN
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import mean_absolute_percentage_error


def prepare_data_onestep(df, col, ann=False):
    train = df.iloc[:-N_TEST]
    test = df.iloc[-N_TEST:]
    train_idx = df.index <= train.index[-1]
    test_idx = df.index > train.index[-1]

    # Make supervised dataset
    series = df[col].dropna().to_numpy()
    try:
        D = series.shape[1]
    except:
        D = 1

    X = []
    Y = []

    for t in range(len(series) - T):
        x = series[t:t + T]
        X.append(x)
        y = series[t + T]
        Y.append(y)

    if ann & D == 1:
        X = np.array(X).reshape(-1, T)
    else:
        X = np.array(X).reshape(-1, T, D)  # For CNN and RNN

    Y = np.array(Y)
    N = len(X)
    print(N, X.shape)

    Xtrain, Ytrain = X[:-N_TEST], Y[:-N_TEST]
    Xtest, Ytest = X[-N_TEST:], Y[-N_TEST:]

    return Xtrain, Ytrain, Xtest, Ytest, train_idx, test_idx


class StocksForecastDL:
    def __init__(self,
                 stock_name_list=('UAL', 'WMT', 'PFE'),
                 start_date='2018-01-01',
                 end_date='2022-12-31',):
        """
        Initialize the StocksForecast class.

        Args:
            stock_name_list (list[str]): List of stock names. Default is ('UAL', 'WMT', 'PFE').
            start_date (str): Start date of the data. Default is '2018-01-01'.
            end_date (str): End date of the data. Default is '2022-12-31'.
        """
        self.dfs = dict()
        for name in stock_name_list:
            self.dfs[name] = yf.download(name, start=start_date, end=end_date)
            self.dfs[name]['Diff'] = self.dfs[name]['Close'].diff(1)
            self.dfs[name]['Log'] = np.log(self.dfs[name]['Close'])

    def run_onestep_forecast(self, stock_name='UAL', col='Log', diff=True):
        df_all = self.dfs[stock_name]
        if diff:
            df_all[f'Diff{col}'] = df_all[col].diff()
            col = f'Diff{col}'

        prepare_data_onestep(df=df_all, col=col, ann=False)

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    N_TEST = 10
    T = 10

    ts = StocksForecastDL()
    ts.run_onestep_forecast()

# # Basic ANN
# i = Input(shape=(T,))
# x = Dense(32, activation='relu')(i)
#
# # RNN
#
# i = Input(shape=(T, 1))
# # can use SimpleRNN/GRU/LSTM
# x = LSTM(32, return_sequences=True)(i) # default is tanh
# x = LSTM(32)(x)
# # when return_sequences=True, can use GlobalMaxPooling1D afterwards
#
# # CNN (1D for time series, 2D for images)
#
# i = Input(shape=(T, 1)) # single value time series
# x = Conv1D(16, 3, activation='relu', padding='same')(i)
# x = MaxPooling1D(2)(x)
# x = Conv1D(32, 3, activation='relu', padding='same')(x)
# x = GlobalMaxPooling1D()(x)
#
# x = Dense(1)(x)
# model = Model(i, x)
#
# model.summary() # CNN and RNN (ANN?)
#
# # change loss for classification and other tasks
# model.compile(
#     loss='mse',
#     optimizer='adam',
#     metrics='mae',
# )
#
# r = model.fit(
#     Xtrain,
#     Ytrain,
#     epochs=100,
#     validation_data=(Xtest, Ytest),
# )
#
# plt.plot(r.history['loss'], label='train loss')
# plt.plot(r.history['val_loss'], label='test loss')
# plt.legend()
# plt.show()
#
# train_idx[:T+1] = False # not predictable
#
# Ptrain = model.predict(Xtrain).flatten()
# Ptest = model.predict(Xtest).flatten()
#
# df.loc[train_idx, 'Diff ANN Train Prediction'] = Ptrain
# df.loc[test_idx, 'Diff ANN Test Prediction'] = Ptest
#
# cols = ['DiffLogPassengers',
#         'Diff Train Prediction',
#         'Diff Test Prediction',]
#
# df[cols].plot(figsize=(15, 5));
#
# # Need to computer un-differenced predictions
# df['ShiftLogPassengers'] = df['LogPassengers'].shift(1)
# prev = df['ShiftLogPassengers']
#
# # Last known train value
# last_train = train.iloc[-1]['LogPassengers']
#
# # 1-step forecast
# df.loc[train_idx, '1step_train'] = prev[train_idx] + Ptrain
# df.loc[test_idx, '1step_test'] = prev[test_idx] + Ptest
#
# col2 = ['LogPassengers',
#         '1step_train',
#         '1step_test',]
# df[col2].plot(figsize=(15, 5));
#
# # multi-step forecast
# multistep_predictions = []
#
# # first test input
# last_x = Xtest[0]
#
# while len(multistep_predictions) < Ntest:
#   # p = model.predict(last_x.reshape(1, -1))[0] # ANN
#   p = model.predict(last_x.reshape(1, -1, 1))[0] # CNN and RNN
#
#   # update the predictions list
#   multistep_predictions.append(p)
#
#   # make the new input
#   last_x = np.roll(last_x, -1)
#   last_x[-1] = p
#
# df.loc[test_idx, 'multistep'] = last_train + np.cumsum(multistep_predictions)
#
# col3 = ['LogPassengers',
#         'multistep',
#         '1step_test',]
# df[col3].plot(figsize=(15, 5))
#
# # make multi-output supervsied dataset
# Tx = T
# Ty = Ntest
# X = []
# Y = []
#
# for t in range(len(series) - Tx - Ty + 1):
#   x = series[t:t+Tx]
#   X.append(x)
#   y = series[t+Tx:t+Tx+Ty]
#   Y.append(y)
#
# # X = np.array(X).reshape(-1, Tx) # ANN
# X = np.array(X).reshape(-1, Tx, 1) # CNN
# Y = np.array(Y).reshape(-1, Ty)
# N = len(X)
# print("Y.shape", Y.shape, "X.shape", X.shape)
#
# Xtrain_m, Ytrain_m = X[:-1], Y[:-1]
# Xtest_m, Ytest_m = X[-1:], Y[-1:]
#
# # Basic ANN
# i = Input(shape=(Tx,))
# x = Dense(32, activation='relu')(i)
#
# # RNN
# i = Input(shape=(Tx, 1))
# x = LSTM(32, return_sequences=True)(i)
# x = LSTM(32)(x)
#
# # CNN
# i = Input(shape=(Tx, 1))
# x = Conv1D(16, 3, activation='relu', padding='same')(i)
# x = MaxPooling1D(2)(x)
# x = Conv1D(32, 3, activation='relu', padding='same')(i)
# x = GlobalMaxPooling1D()(x)
#
# x = Dense(Ty)(x)
# model = Model(i, x)
#
# model.summary()
#
# check_point = ModelCheckpoint(
#     'best_model.h5', monitor='val_loss', save_best_only=True
# )
#
# model.compile(
#     loss='mse',
#     optimizer='adam',
#     metrics='mae',
# )
#
# r = model.fit(
#     Xtrain_m,
#     Ytrain_m,
#     epochs=100,
#     validation_data=(Xtest_m, Ytest_m),
#     callbacks=[check_point],
# )
#
# plt.plot(r.history['loss'], label='train loss')
# plt.plot(r.history['val_loss'], label='test lsos')
# plt.legend()
# plt.show()
#
# best_model = tf.keras.models.load_model('best_model.h5')
#
# Ptrain = model.predict(Xtrain_m)
# Ptest = model.predict(Xtest_m)
# print(Ptrain.shape, Ptest.shape)
#
# Ptrain = Ptrain[:,0] # prediction for 1 stemp ahead (zeroth row)
# Ptest = Ptest[0]
#
# df.loc[test_idx, 'Diff Multi-Output Test Prediction'] = Ptest
# col5 = ['DiffLogPassengers', 'Diff Multi-Output Test Prediction']
# df[col5].plot(figsize=(15, 5));
#
# df.loc[test_idx, 'multioutput'] = last_train + np.cumsum(Ptest)
#
# col4 = ['LogPassengers', 'multistep', '1step_test', 'multioutput']
# df[col4].plot(figsize=(15, 5));
#
# # MAPE
# test_log_pass = df.iloc[-Ntest:]['LogPassengers']
# mape1 = mean_absolute_percentage_error(
#     test_log_pass, df.loc[test_idx, 'multistep']
# )
# print("multi-step MAPE:", mape1)
# mape2 = mean_absolute_percentage_error(
#     test_log_pass, df.loc[test_idx, 'multioutput']
# )
# print("multi-output MAPE:", mape2)