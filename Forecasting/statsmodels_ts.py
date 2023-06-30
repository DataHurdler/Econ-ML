import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import VAR
import pmdarima as pm

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")  # ignore warnings

# TODO:BUILD THE WALK-FORWARD VALIDATION SO IT WORKS FOR ALL METHODS
# Other things that may be of interest:
# boxcox: from scipy.stats import boxcox
# Test for stationarity: from statsmodels.tsa.stattools import adfuller
# VARMAX: from statsmodels.tsa.statespace.varmax import VARMAX
# ARIMA: from statsmodels.tsa.arima.model import ARIMA


def prepare_data(df):
    """
    Split the data into training and testing sets.

    Args:
        df (pandas.DataFrame): The input dataframe.

    Returns:
        tuple: A tuple containing the train set, test set, train index, and test index.
    """
    train = df.iloc[:-N_TEST]
    test = df.iloc[-N_TEST:]
    train_idx = df.index <= train.index[-1]
    test_idx = df.index > train.index[-1]

    return train, test, train_idx, test_idx


def plot_fitted_forecast(df, train_idx, test_idx, model_result, col=None, forecast_df=None, arima=False):
    """
    Plot the fitted and forecasted values of a time series.

    Args:
        df (pandas.DataFrame): The input dataframe.
        train_idx (numpy.ndarray): Boolean array indicating the train index.
        test_idx (numpy.ndarray): Boolean array indicating the test index.
        model_result: The fitted model or forecast model result.
        col (str): The column name to plot. Default is None.
        forecast_df (pandas.DataFrame): The forecasted dataframe. Default is None.
        arima (bool): Whether the model is ARIMA or not. Default is False.
    """
    if arima:
        df.loc[train_idx, 'fitted'] = model_result.predict_in_sample(end=-1)
        df.loc[test_idx, 'forecast'] = np.array(model_result.predict(n_periods=N_TEST, return_conf_int=False))
    elif forecast_df is None:
        df.loc[train_idx, 'fitted'] = model_result.fittedvalues
        df.loc[test_idx, 'forecast'] = np.array(model_result.forecast(N_TEST))
    else:
        col = "Scaled_" + col
        df.loc[train_idx, 'fitted'] = model_result.fittedvalues[col]
        df.loc[test_idx, 'forecast'] = forecast_df[col].values

    df = df[-108:]  # only plot the last 108 days

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df.index, df[f"{col}"], label='data')
    ax.plot(df.index, df['fitted'], label='fitted')
    ax.plot(df.index, df['forecast'], label='forecast')

    plt.legend()
    plt.show()


class StocksForecast:
    def __init__(self, stock_name_list=('UAL', 'WMT', 'PFE'), start_date='2018-01-01', end_date='2022-12-31'):
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

    def run_ets(self, stock_name='UAL', col='Close'):
        """
        Run the Exponential Smoothing (ETS) model on the specified stock.

        Args:
            stock_name (str): The name of the stock. Default is 'UAL'.
            col (str): The column name to use for the model. Default is 'Close'.
        """
        train, test, train_idx, test_idx = prepare_data(self.dfs[stock_name])

        model = ExponentialSmoothing(train[col].dropna(), trend='mul', seasonal='mul', seasonal_periods=252)
        result = model.fit()

        plot_fitted_forecast(self.dfs[stock_name], train_idx, test_idx, result, col)

    def walkforward(self, h, steps, tuple_of_option_lists, stock_name='UAL', col='Close', debug=False):
        """
        Perform walk-forward validation on the specified stock. Only supports ExponentialSmoothing

        Args:
            h (int): The forecast horizon.
            steps (int): The number of steps to walk forward.
            tuple_of_option_lists (tuple): Tuple of option lists for trend and seasonal types.
            stock_name (str): The name of the stock. Default is 'UAL'.
            col (str): The column name to use for the model. Default is 'Close'.
            debug (bool): Whether to print debug information. Default is False.

        Returns:
            float: The mean of squared errors.
        """
        errors = []
        seen_last = False
        steps_completed = 0
        df = self.dfs[stock_name]
        Ntest = len(df) - h - steps + 1

        trend_type, seasonal_type = tuple_of_option_lists

        for end_of_train in range(Ntest, len(df) - h + 1):
            train = df.iloc[:end_of_train]
            test = df.iloc[end_of_train:end_of_train + h]

            if test.index[-1] == df.index[-1]:
                seen_last = True

            steps_completed += 1

            hw = ExponentialSmoothing(train[col], trend=trend_type, seasonal=seasonal_type, seasonal_periods=40)

            result_hw = hw.fit()

            forecast = result_hw.forecast(h)
            error = mean_squared_error(test[col], np.array(forecast))
            errors.append(error)

        if debug:
            print("seen_last:", seen_last)
            print("steps completed:", steps_completed)

        return np.mean(errors)

    def run_var(self, stock_list=('UAL', 'WMT', 'PFE'), col='Close'):
        """
        Run the Vector Autoregression (VAR) model on the specified stocks.

        Args:
            stock_list (tuple): Tuple of stock names. Default is ('UAL', 'WMT', 'PFE').
            col (str): The column name to use for the model. Default is 'Close'.
        """
        df_all = pd.DataFrame(index=self.dfs[stock_list[0]].index)
        for stock in stock_list:
            df_all = df_all.join(self.dfs[stock][col].dropna())
            df_all.rename(columns={col: f"{stock}_{col}"}, inplace=True)

        train, test, train_idx, test_idx = prepare_data(df_all)

        stock_cols = df_all.columns.values

        for value in stock_cols:
            scaler = StandardScaler()
            train[f'Scaled_{value}'] = scaler.fit_transform(train[[value]])
            test[f'Scaled_{value}'] = scaler.transform(test[[value]])
            df_all.loc[train_idx, f'Scaled_{value}'] = train[f'Scaled_{value}']
            df_all.loc[test_idx, f'Scaled_{value}'] = test[f'Scaled_{value}']

        cols = ['Scaled_' + value for value in stock_cols]

        plot_acf(train[cols[0]])
        plot_pacf(train[cols[0]])
        plot_pacf(train[cols[-1]])

        model = VAR(train[cols])
        result = model.fit(maxlags=40, method='mle', ic='aic')
        lag_order = result.k_ar

        prior = train.iloc[-lag_order:][cols].to_numpy()
        forecast_df = pd.DataFrame(result.forecast(prior, N_TEST), columns=cols)

        plot_fitted_forecast(df_all, train_idx, test_idx, result, stock_cols[0], forecast_df)

        df_all.loc[train_idx, 'fitted'] = result.fittedvalues[cols[0]]
        df_all.loc[test_idx, 'forecast'] = forecast_df[cols[0]].values

        train_pred = df_all.loc[train_idx, 'fitted'].iloc[lag_order:]
        train_true = df_all.loc[train_idx, cols[0]].iloc[lag_order:]

        print("VAR Train R2: ", r2_score(train_true, train_pred))

        test_pred = df_all.loc[test_idx, 'forecast']
        test_true = df_all.loc[test_idx, cols[0]]
        print("VAR Test R2:", r2_score(test_true, test_pred))

    def run_arima(self, stock_name='UAL', col='Close', seasonal=True, m=12):
        """
        Run the Auto Autoregressive Integrated Moving Average (ARIMA) model on the specified stock.

        Args:
            stock_name (str): The name of the stock. Default is 'UAL'.
            col (str): The column name to use for the model. Default is 'Close'.
            seasonal (bool): Whether to include seasonal components. Default is True.
            m (int): The number of periods in each seasonal cycle. Default is 12.
        """
        train, test, train_idx, test_idx = prepare_data(self.dfs[stock_name])

        model = pm.auto_arima(train[col], trace=True, suppress_warnings=True, seasonal=seasonal, m=m)

        print(model.summary())

        plot_fitted_forecast(self.dfs[stock_name], train_idx, test_idx, model, col, arima=True)


if __name__ == "__main__":
    N_TEST = 10
    H = 20  # 4 weeks
    STEPS = 10

    # Hyperparameters to try
    trend_type_list = ['add', 'mul']
    seasonal_type_list = ['add', 'mul']
    init_method_list = ['estimated', 'heuristic', 'legacy-heristic']
    use_boxcox_list = [True, False, 0]

    ts = StocksForecast()

    ts.run_ets(stock_name='UAL', col='Log')
    ts.run_var(col='Log')
    ts.run_arima(stock_name='UAL', col='Log')

    tuple_of_option_lists = (trend_type_list, seasonal_type_list,)
    best_score = float('inf')
    best_options = None

    for x in itertools.product(*tuple_of_option_lists):
        score = ts.walkforward(h=H, steps=STEPS, stock_name='UAL', col='Log', tuple_of_option_lists=x)

        if score < best_score:
            print("Best score so far:", score)
            best_score = score
            best_options = x

    trend_type, seasonal_type = best_options
    print(f"best trend type: {trend_type}")
    print(f"best seasonal type: {seasonal_type}")
