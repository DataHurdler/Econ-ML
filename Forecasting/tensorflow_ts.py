# import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
import sys

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input  # ANN
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN  # RNN
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D  # CNN
from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import mean_absolute_percentage_error


# TODO:Support for multiple inputs/outputs


def ann(T, num_layers=32):
    # Basic ANN
    i = Input(shape=(T,))
    x = Dense(num_layers, activation='relu')(i)
    x = Dense(num_layers, activation='relu')(x)

    return i, x


def rnn(T, D, num_layers=32, rnn_model="lstm"):
    i = Input(shape=(T, D))
    # can use SimpleRNN/GRU/LSTM
    if rnn_model == 'lstm':
        x = LSTM(num_layers, return_sequences=True)(i)  # default is tanh
        x = LSTM(num_layers, )(x)
    elif rnn_model == 'gru':
        x = GRU(num_layers, return_sequences=True)(i)  # default is tanh
        x = GRU(num_layers, )(x)
    else:
        x = SimpleRNN(num_layers, return_sequences=True)(i)  # default is tanh
        x = SimpleRNN(num_layers, )(x)
    # when return_sequences=True, can use GlobalMaxPooling1D after wards

    return i, x


def cnn(T, D):
    # CNN (1D for time series, 2D for images)
    i = Input(shape=(T, D))  # single value time series
    x = Conv1D(16, 3, activation='relu', padding='same')(i)
    x = MaxPooling1D(2)(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = GlobalMaxPooling1D()(x)

    return i, x


class StocksForecastDL:
    def __init__(self,
                 stock_name_list=('UAL', 'WMT', 'PFE'),
                 start_date='2018-01-01',
                 end_date='2022-12-31',
                 t=10,
                 n_test=12,
                 epochs=200, ):
        """
        Initialize the StocksForecast class.

        Args:
            stock_name_list (list[str]): List of stock names. Default is ('UAL', 'WMT', 'PFE').
            start_date (str): Start date of the data. Default is '2018-01-01'.
            end_date (str): End date of the data. Default is '2022-12-31'.
        """
        self.T = t
        self.N_TEST = n_test
        self.EPOCHS = epochs
        self.dfs = dict()
        for name in stock_name_list:
            self.dfs[name] = yf.download(name, start=start_date, end=end_date)
            self.dfs[name]['Diff'] = self.dfs[name]['Close'].diff(1)
            self.dfs[name]['Log'] = np.log(self.dfs[name]['Close'])

        self.train_idx = []
        self.test_idx = []

    def prepare_data(self, df, col, ann=False, multistep=False):
        train = df.iloc[:-self.N_TEST]
        test = df.iloc[-self.N_TEST:]
        train_idx = df.index <= train.index[-1]
        test_idx = df.index > train.index[-1]

        # Make supervised dataset
        series = df[col].dropna().to_numpy()
        try:
            d = series.shape[1]
        except AttributeError:
            d = 1

        X = []
        Y = []

        start_idx = self.N_TEST
        if multistep:
            for t in range(len(series) - self.T - self.N_TEST + 1):
                x = series[t:t + self.T]
                X.append(x)
                y = series[t + self.T:t + self.T + self.N_TEST]
                Y.append(y)

            Y = np.array(Y).reshape(-1, self.N_TEST)
            start_idx = 1
        else:
            for t in range(len(series) - self.T):
                x = series[t:t + self.T]
                X.append(x)
                y = series[t + self.T]
                Y.append(y)

            Y = np.array(Y)

        if ann and d == 1:
            X = np.array(X).reshape(-1, self.T)
        else:
            X = np.array(X).reshape(-1, self.T, d)  # For CNN and RNN

        N = len(X)

        Xtrain, Ytrain = X[:-start_idx], Y[:-start_idx]
        Xtest, Ytest = X[-start_idx:], Y[-start_idx:]

        return Xtrain, Ytrain, Xtest, Ytest, train_idx, test_idx, N, d

    def make_predictions(self, df, orig_col, train_idx, test_idx, Xtrain, Xtest, model, ann=False, multistep=False):
        train = df.iloc[:-self.N_TEST]
        train_idx[:self.T + 1] = False  # not predictable

        if multistep:
            Ptrain = model.predict(Xtrain)[:, 0]
            Ptest = model.predict(Xtest)[0]
        else:
            Ptrain = model.predict(Xtrain).flatten()
            Ptest = model.predict(Xtest).flatten()

        # Need to computer un-differenced predictions
        for c in orig_col:
            df[f'Shift{c}'] = df[c].shift(1)

        new_col = ["Shift" + word for word in orig_col]
        prev = df[new_col]

        # Last known train value
        last_train = train.iloc[-1][orig_col]

        if not multistep:
            # 1-step forecast
            df.loc[train_idx, '1step_train'] = prev[train_idx].squeeze() + Ptrain
            df.loc[test_idx, '1step_test'] = prev[test_idx].squeeze() + Ptest

            col2 = ['1step_train',
                    '1step_test',
                    ]
            df[col2].plot(figsize=(15, 5))
            plt.show()

            # multi-step forecast for single step model
            multistep_predictions = []

            # first test input
            last_x = Xtest[0]

            while len(multistep_predictions) < self.N_TEST:
                if ann:
                    p = model.predict(last_x.reshape(1, -1))[0]  # ANN
                else:
                    p = model.predict(last_x.reshape(1, -1, 1))[0]  # CNN and RNN

                # update the predictions list
                multistep_predictions.append(p)

                # make the new input
                last_x = np.roll(last_x, -1)
                last_x[-1] = p[0]

            df.loc[test_idx, 'multistep'] = last_train[0] + np.cumsum(multistep_predictions)

            col3 = ['multistep',
                    '1step_test',
                    ]
            df[col3].plot(figsize=(15, 5))
            plt.show()

        else:
            df.loc[test_idx, 'multioutput'] = last_train[0] + np.cumsum(Ptest)

    def run_forecast(self,
                     stock_name: str = 'UAL',
                     col: list = ['Log'],
                     diff=True,
                     model="cnn",
                     multistep=False,
                     **kwargs):
        df_all = self.dfs[stock_name]

        D = len(col)
        if D > 1 and model == "ann":
            warnings.warn("Currently, ANN only runs with a single variable.")
            sys.exit(1)

        new_col = col.copy()
        if diff:
            for c in col:
                df_all[f'Diff{c}'] = df_all[c].diff()
            new_col = ["Diff" + word for word in new_col]

        if model == 'ann':
            ann_bool = True
        else:
            ann_bool = False

        output = self.prepare_data(df=df_all, col=new_col, ann=ann_bool, multistep=multistep)
        Xtrain, Ytrain, Xtest, Ytest, train_idx, test_idx, N, D = output

        self.train_idx = train_idx
        self.test_idx = test_idx

        build_model = eval(model)
        if model == 'ann':
            i, x = build_model(T=self.T, **kwargs)
        else:
            i, x = build_model(T=self.T, D=D, **kwargs)

        if multistep:
            x = Dense(self.N_TEST)(x)
        else:
            x = Dense(1)(x)

        model = Model(i, x)

        model.summary()  # CNN and RNN (ANN?)

        # change loss for classification and other tasks
        # BinaryCrossentropy(from_logits=True)
        # SparseCategoricalCrossentropy(from_logits=True)
        model.compile(
            loss='mse',
            optimizer='adam',
            metrics='mae',
        )

        r = model.fit(
            Xtrain,
            Ytrain,
            epochs=self.EPOCHS,
            validation_data=(Xtest, Ytest),
            verbose=0,
        )

        plt.plot(r.history['loss'], label='train loss')
        plt.plot(r.history['val_loss'], label='test loss')
        plt.legend()
        plt.show()

        self.make_predictions(df_all, col, train_idx, test_idx, Xtrain, Xtest, model, ann_bool, multistep)

    def single_model_comparison(self,
                                stock_name: str = 'UAL',
                                col: list = ['Log'],
                                diff=True,
                                model="cnn",
                                **kwargs):

        df_all = self.dfs[stock_name]

        self.run_forecast(model=model, diff=diff, **kwargs)
        self.run_forecast(model=model, diff=diff, multistep=True, **kwargs)

        pred_cols = col + ['multistep', '1step_test', 'multioutput']
        df_all[pred_cols][-(self.N_TEST * 3):].plot(figsize=(15, 5))
        plt.show()

        # MAPE
        test_log_pass = df_all.iloc[-self.N_TEST:][col]
        mape1 = mean_absolute_percentage_error(
            test_log_pass, df_all.loc[self.test_idx, 'multistep']
        )
        print("multi-step MAPE:", mape1)
        mape2 = mean_absolute_percentage_error(
            test_log_pass, df_all.loc[self.test_idx, 'multioutput']
        )
        print("multi-output MAPE:", mape2)

        return df_all


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    plt.ion()

    ts = StocksForecastDL(t=20, epochs=100)

    ts.single_model_comparison(model="ann", diff=True)
    ts.single_model_comparison(model="cnn", diff=True)
    ts.single_model_comparison(model="rnn", rnn_model="lstm", diff=True)
