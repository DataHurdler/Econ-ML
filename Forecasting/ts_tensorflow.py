import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
import sys
import random

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input  # ANN
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN  # RNN
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D  # CNN
from tensorflow.keras.models import Model


def ann(T, num_units=32):
    """
    Create a basic Artificial Neural Network (ANN) model.

    Args:
        T (int): Time steps or input size.
        num_layers (int): Number of layers in the ANN.

    Returns:
        tuple: Input and output tensors of the ANN model.
    """
    i = Input(shape=(T,))
    x = Dense(num_units, activation='relu')(i)
    x = Dense(num_units, activation='relu')(x)

    return i, x


def rnn(T, D, num_units=32, rnn_model="lstm"):
    """
    Create a Recurrent Neural Network (RNN) model.

    Args:
        T (int): Time steps or input size.
        D (int): Dimensionality of the input.
        num_layers (int): Number of layers in the RNN.
        rnn_model (str): RNN model type ('lstm', 'gru', or 'simple_rnn').

    Returns:
        tuple: Input and output tensors of the RNN model.
    """
    i = Input(shape=(T, D))
    if rnn_model == 'lstm':
        x = LSTM(num_units, return_sequences=True)(i)
        x = LSTM(num_units)(x)
    elif rnn_model == 'gru':
        x = GRU(num_units, return_sequences=True)(i)
        x = GRU(num_units)(x)
    else:
        x = SimpleRNN(num_units, return_sequences=True)(i)
        x = SimpleRNN(num_units)(x)

    return i, x


def cnn(T, D):
    """
    Create a Convolutional Neural Network (CNN) model.

    Args:
        T (int): Time steps or input size.
        D (int): Dimensionality of the input.

    Returns:
        tuple: Input and output tensors of the CNN model.
    """
    i = Input(shape=(T, D))
    x = Conv1D(16, 3, activation='relu', padding='same')(i)
    x = MaxPooling1D(2)(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = GlobalMaxPooling1D()(x)

    return i, x


class StocksForecastDL:
    def __init__(
        self,
        stock_name_list=('UAL', 'WMT', 'PFE'),
        start_date='2018-01-01',
        end_date='2022-12-31',
        t=10,
        n_test=12,
        epochs=200,
    ):
        """
        Initialize the StocksForecastDL class.

        Args:
            stock_name_list (tuple): List of stock names.
            start_date (str): Start date for data retrieval.
            end_date (str): End date for data retrieval.
            t (int): Number of time steps.
            n_test (int): length of forecast horizon.
            epochs (int): Number of training epochs.
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

    def prepare_data(self, df, col, ann=False, multioutput=False):
        """
        Prepare the data for training and testing.

        Args:
            df (DataFrame): Input data.
            col (list): List of columns to be used.
            ann (bool): Indicates whether ANN model is used.
            multioutput (bool): Indicates whether multioutput prediction is performed.

        Returns:
            tuple: Prepared data for training and testing.
        """
        train = df.iloc[:-self.N_TEST]
        test = df.iloc[-self.N_TEST:]
        train_idx = df.index <= train.index[-1]
        test_idx = df.index > train.index[-1]

        series = df[col].dropna().to_numpy()
        try:
            d = series.shape[1]
        except AttributeError:
            d = 1

        X = []
        Y = []

        start_idx = self.N_TEST
        if multioutput:
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
            X = np.array(X).reshape(-1, self.T, d)

        N = len(X)

        Xtrain, Ytrain = X[:-start_idx], Y[:-start_idx]
        Xtest, Ytest = X[-start_idx:], Y[-start_idx:]

        return Xtrain, Ytrain, Xtest, Ytest, train_idx, test_idx, N, d

    def make_predictions(self,
                         stock_name,
                         orig_col,
                         train_idx,
                         test_idx,
                         Xtrain,
                         Xtest,
                         model,
                         ann=False,
                         multioutput=False):
        """
        Make predictions using the trained model.

        Args:
            stock_name (str): Name of the stock.
            orig_col (list): Original columns used for predictions.
            train_idx (bool): Index of training samples.
            test_idx (bool): Index of test samples.
            Xtrain (ndarray): Training input data.
            Xtest (ndarray): Test input data.
            model (Model): Trained model.
            ann (bool): Indicates whether ANN model is used.
            multioutput (bool): Indicates whether multioutput prediction is performed.
        """
        train = self.dfs[stock_name].iloc[:-self.N_TEST]
        train_idx[:self.T + 1] = False

        if multioutput:
            Ptrain = model.predict(Xtrain)[:, 0]
            Ptest = model.predict(Xtest)[0]
        else:
            Ptrain = model.predict(Xtrain).flatten()
            Ptest = model.predict(Xtest).flatten()

        for c in orig_col:
            self.dfs[stock_name][f'Shift{c}'] = self.dfs[stock_name][c].shift(1)

        new_col = ["Shift" + word for word in orig_col]
        prev = self.dfs[stock_name][new_col]

        last_train = train.iloc[-1][orig_col]

        if not multioutput:
            self.dfs[stock_name].loc[train_idx, '1step_train'] = prev[train_idx].squeeze() + Ptrain
            self.dfs[stock_name].loc[test_idx, '1step_test'] = prev[test_idx].squeeze() + Ptest

            multistep_predictions = []
            last_x = Xtest[0]

            while len(multistep_predictions) < self.N_TEST:
                if ann:
                    p = model.predict(last_x.reshape(1, -1))[0]
                else:
                    p = model.predict(last_x.reshape(1, -1, 1))[0]

                multistep_predictions.append(p)
                last_x = np.roll(last_x, -1)
                last_x[-1] = p[0]

            self.dfs[stock_name].loc[test_idx, 'multistep'] = last_train[0] + np.cumsum(multistep_predictions)

        else:
            self.dfs[stock_name].loc[test_idx, 'multioutput'] = last_train[0] + np.cumsum(Ptest)

    def run_forecast(self,
                     stock_name: str = 'UAL',
                     col: list = ['Log'],
                     diff=True,
                     model="cnn",
                     multioutput=False,
                     plot=True,
                     **kwargs):
        """
        Run the forecast for a given stock.

        Args:
            stock_name (str): Name of the stock.
            col (list): List of columns to be used.
            diff (bool): Indicates whether differencing is applied.
            model (str): Model type ('ann', 'rnn', or 'cnn').
            multioutput (bool): Indicates whether multioutput prediction is performed.
            plot (bool): Whether to plot. Default is True.
            **kwargs: Additional keyword arguments for the selected model.
        """

        D = len(col)
        if D > 1 and model == "ann":
            warnings.warn("Currently, ANN only runs with a single variable.")
            sys.exit(1)

        new_col = col.copy()
        if diff:
            for c in col:
                self.dfs[stock_name][f'Diff{c}'] = self.dfs[stock_name][c].diff()
            new_col = ["Diff" + word for word in new_col]

        if model == 'ann':
            ann_bool = True
        else:
            ann_bool = False

        output = self.prepare_data(df=self.dfs[stock_name], col=new_col, ann=ann_bool, multioutput=multioutput)
        Xtrain, Ytrain, Xtest, Ytest, train_idx, test_idx, N, D = output

        self.train_idx = train_idx
        self.test_idx = test_idx

        build_model = eval(model)
        if model == 'ann':
            i, x = build_model(T=self.T, **kwargs)
        else:
            i, x = build_model(T=self.T, D=D, **kwargs)

        if multioutput:
            x = Dense(self.N_TEST)(x)
        else:
            x = Dense(1)(x)

        nn_model = Model(i, x)

        nn_model.summary()

        nn_model.compile(
            loss='mse',
            optimizer='adam',
            metrics='mae',
        )

        r = nn_model.fit(
            Xtrain,
            Ytrain,
            epochs=self.EPOCHS,
            validation_data=(Xtest, Ytest),
            verbose=0,
        )

        if plot:
            plt.figure(figsize=(15, 5))
            plt.plot(r.history['loss'], label='train loss')
            plt.plot(r.history['val_loss'], label='test loss')
            plt.legend()

            if multioutput:
                step_type = "multi"
            else:
                step_type = "single"
            if model == "rnn":
                rnn_model = kwargs.get("rnn_model")
                plt.savefig(f"{model}_{rnn_model}_{step_type}_hist.png", dpi=300)
            else:
                plt.savefig(f"{model}_{step_type}_hist.png", dpi=300)
            plt.clf()

        self.make_predictions(stock_name, col, train_idx, test_idx, Xtrain, Xtest, nn_model, ann_bool, multioutput)

    def single_model_comparison(self,
                                stock_name: str = 'UAL',
                                col: list = ['Log'],
                                diff=True,
                                model="cnn",
                                plot=True,
                                **kwargs):
        """
        Perform a comparison of a single model for a given stock.

        Args:
            stock_name (str): Name of the stock.
            col (list): List of columns to be used.
            diff (bool): Indicates whether differencing is applied.
            model (str): Model type ('ann', 'rnn', or 'cnn').
            plot (bool): Whether to plot. Default is True.
            **kwargs: Additional keyword arguments for the selected model.

        Returns:
            DataFrame: DataFrame containing the predictions and evaluation metrics.
        """

        self.run_forecast(model=model, stock_name=stock_name, diff=diff, plot=plot, **kwargs)
        self.run_forecast(model=model, stock_name=stock_name, diff=diff, multioutput=True, plot=plot, **kwargs)

        if plot:
            pred_cols = col + ['1step_test', 'multistep', 'multioutput']
            self.dfs[stock_name][pred_cols][-(self.N_TEST * 3):].plot(figsize=(15, 5))

            if model == "rnn":
                rnn_model = kwargs.get("rnn_model")
                plt.savefig(f"{model}_{rnn_model}_comparison.png", dpi=300)
            else:
                plt.savefig(f"{model}_comparison.png", dpi=300)
            plt.clf()


if __name__ == "__main__":
    # make sure to get reproducible results
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    plt.ion()

    ts = StocksForecastDL(t=20, epochs=1500)

    ts.single_model_comparison(model="ann", diff=True)
    ts.single_model_comparison(model="cnn", diff=True)
    ts.single_model_comparison(model="rnn", rnn_model="lstm", diff=True)
