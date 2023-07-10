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


def plot_fitted_forecast(df, name, col=None):
    """
    Plot the fitted and forecasted values of a time series.

    Args:
        df (pandas.DataFrame): The input dataframe.
        name (str): Name of model that is being plotted.
        col (str): The column name to plot. Default is None.
    """
    df = df[-54:]  # only plot the last 54 days

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df.index, df[f"{col}"], label='data')
    ax.plot(df.index, df[f"{name}_fitted"], label='fitted')
    ax.plot(df.index, df[f"{name}_forecast"], label='forecast')

    plt.legend()
    plt.savefig(f"statsmodels_{name}.png", dpi=300)


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

        self.result_for_plot = pd.DataFrame(index=self.dfs[stock_name_list[0]].index)

    def run_ets(self, stock_name='UAL', col='Close'):
        """
        Run the Exponential Smoothing (ETS) model on the specified stock.

        Args:
            stock_name (str): The name of the stock. Default is 'UAL'.
            col (str): The column name to use for the model. Default is 'Close'.
        """
        name = 'ets'

        df_all = self.dfs[stock_name]
        train, test, train_idx, test_idx = prepare_data(df_all)

        model = ExponentialSmoothing(train[col].dropna(), trend='mul', seasonal='mul', seasonal_periods=252)
        result = model.fit()

        df_all.loc[train_idx, f"{name}_fitted"] = result.fittedvalues
        df_all.loc[test_idx, f"{name}_forecast"] = np.array(result.forecast(N_TEST))

        self.result_for_plot = self.result_for_plot.merge(df_all[[f"{name}_fitted", f"{name}_forecast"]],
                                                          left_index=True,
                                                          right_index=True)

        plot_fitted_forecast(df_all, 'ets', col)

    def walkforward_ets(self, h, steps, tuple_of_option_lists, stock_name='UAL', col='Close', debug=False):
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

    def run_walkforward(self, h, steps, stock_name, col, options):
        """
            Perform walk-forward validation on the specified stock using Exponential Smoothing (ETS).

            Args:
                h (int): The forecast horizon.
                steps (int): The number of steps to walk forward.
                stock_name (str): The name of the stock.
                col (str): The column name to use for the model.
                options (tuple): Tuple of option lists for trend and seasonal types.

            Returns:
                float: The mean squared error (MSE) of the forecast.
        """
        best_score = float('inf')
        best_options = None

        for x in itertools.product(*options):
            score = self.walkforward_ets(h=h, steps=steps, stock_name=stock_name, col=col, tuple_of_option_lists=x)

            if score < best_score:
                print("Best score so far:", score)
                best_score = score
                best_options = x

        trend_type, seasonal_type = best_options
        print(f"best trend type: {trend_type}")
        print(f"best seasonal type: {seasonal_type}")

    def prepare_data_var(self, stock_list, col):
        """
            Prepare the data for Vector Autoregression (VAR) modeling.

            Args:
                stock_list (list): List of stock names.
                col (str): The column name to use for the model.

            Returns:
                tuple: A tuple containing the combined dataframe, train set, test set, train index,
                       test index, stock columns, and scaled columns.
        """
        df_all = pd.DataFrame(index=self.dfs[stock_list[0]].index)
        for stock in stock_list:
            df_all = df_all.join(self.dfs[stock][col].dropna())
            df_all.rename(columns={col: f"{stock}_{col}"}, inplace=True)

        train, test, train_idx, test_idx = prepare_data(df_all)

        stock_cols = df_all.columns.values

        # save all scalers into a list for inverse_transform fitted values and forecast
        scaler_list = list()
        scaler_idx = 0

        # standardizing different stocks
        for value in stock_cols:
            scaler = StandardScaler()
            scaler_list.append(scaler)
            scaler_idx += 1

            train[f'Scaled_{value}'] = scaler.fit_transform(train[[value]])
            test[f'Scaled_{value}'] = scaler.transform(test[[value]])
            df_all.loc[train_idx, f'Scaled_{value}'] = train[f'Scaled_{value}']
            df_all.loc[test_idx, f'Scaled_{value}'] = test[f'Scaled_{value}']

        cols = ['Scaled_' + value for value in stock_cols]

        return df_all, train, test, train_idx, test_idx, stock_cols, cols, scaler_list

    def run_var(self, stock_list=('UAL', 'WMT', 'PFE'), col='Close'):
        """
        Run the Vector Autoregression (VAR) model on the specified stocks.

        Args:
            stock_list (tuple): Tuple of stock names. Default is ('UAL', 'WMT', 'PFE').
            col (str): The column name to use for the model. Default is 'Close'.
        """

        name = 'var'
        output = self.prepare_data_var(stock_list, col)
        df_all, train, test, train_idx, test_idx, stock_cols, cols, scaler_list = output

        model = VAR(train[cols])
        result = model.fit(maxlags=40, method='mle', ic='aic')

        lag_order = result.k_ar
        prior = train.iloc[-lag_order:][cols].to_numpy()
        forecast_df = pd.DataFrame(result.forecast(prior, N_TEST), columns=cols)

        train_idx[:lag_order] = False

        # index "0" for the first stock (UAL). Can modify to be more generic
        scaler = scaler_list[0]
        fitted_scaled = result.fittedvalues[cols[0]]
        fitted = scaler.inverse_transform(np.reshape(fitted_scaled, (-1, 1)))
        forecast_scaled = forecast_df[cols[0]].values
        forecast = scaler.inverse_transform(np.reshape(forecast_scaled, (-1, 1)))

        df_all.loc[train_idx, f"{name}_fitted"] = fitted
        df_all.loc[test_idx, f"{name}_forecast"] = forecast

        self.result_for_plot = self.result_for_plot.merge(df_all[[f"{name}_fitted", f"{name}_forecast"]],
                                                          left_index=True,
                                                          right_index=True)

        col = stock_cols[0]
        plot_fitted_forecast(df_all, 'var', col)

        # Calculate R2
        print("VAR Train R2: ", r2_score(df_all.loc[train_idx, cols[0]].iloc[lag_order:],
                                         df_all.loc[train_idx, f"{name}_fitted"].iloc[lag_order:]))
        print("VAR Test R2: ", r2_score(df_all.loc[test_idx, cols[0]],
                                        df_all.loc[test_idx, f"{name}_forecast"]))

    def run_arima(self, stock_name='UAL', col='Close', seasonal=True, m=12):
        """
        Run the Auto Autoregressive Integrated Moving Average (ARIMA) model on the specified stock.

        Args:
            stock_name (str): The name of the stock. Default is 'UAL'.
            col (str): The column name to use for the model. Default is 'Close'.
            seasonal (bool): Whether to include seasonal components. Default is True.
            m (int): The number of periods in each seasonal cycle. Default is 12.
        """
        name = 'arima'
        df_all = self.dfs[stock_name]
        train, test, train_idx, test_idx = prepare_data(df_all)

        plot_acf(train[col])
        plot_pacf(train[col])

        model = pm.auto_arima(train[col], trace=True, suppress_warnings=True, seasonal=seasonal, m=m)

        print(model.summary())

        df_all.loc[train_idx, f"{name}_fitted"] = model.predict_in_sample(end=-1)
        df_all.loc[test_idx, f"{name}_forecast"] = np.array(model.predict(n_periods=N_TEST, return_conf_int=False))

        self.result_for_plot = self.result_for_plot.merge(df_all[[f"{name}_fitted", f"{name}_forecast"]],
                                                          left_index=True,
                                                          right_index=True)

        plot_fitted_forecast(df_all, 'arima', col)

    def plot_result_comparison(self, stock_name='UAL', col='Log'):
        self.result_for_plot = self.result_for_plot.merge(self.dfs[stock_name][col],
                                                          left_index=True,
                                                          right_index=True)

        colors = ['red', 'red', 'blue', 'blue', 'orange', 'orange', 'pink']
        line_styles = [':', '--', ':', '--', ':', '--', '-']
        legend_names = ['ETS Fitted Values', 'ETS Forecast',
                        'VAR Fitted Values', 'VAR Forecast',
                        'ARIMA Fitted Values', 'ARIMA Forecast',
                        'Data (in log)']

        self.result_for_plot[-27:].plot(figsize=(15, 5),
                                        color=colors, style=line_styles, legend=True)
        plt.legend(labels=legend_names)
        plt.savefig("comparison.png", dpi=300)


if __name__ == "__main__":
    # parameters
    STOCK = 'UAL'
    COL = 'Log'
    N_TEST = 10
    H = 20  # 4 weeks
    STEPS = 10

    # Hyperparameters to try in ETS walk-forward validation
    trend_type_list = ['add', 'mul']
    seasonal_type_list = ['add', 'mul']
    init_method_list = ['estimated', 'heuristic', 'legacy-heristic']  # not used
    use_boxcox_list = [True, False, 0]  # not used

    ts = StocksForecast()

    ts.run_ets(stock_name=STOCK, col=COL)
    ts.run_var(col=COL)
    ts.run_arima(stock_name=STOCK, col=COL)

    ts.plot_result_comparison()

    tuple_of_option_lists = (trend_type_list, seasonal_type_list,)
    ts.run_walkforward(H, STEPS, STOCK, COL, tuple_of_option_lists)

# TODO:BUILD THE WALK-FORWARD VALIDATION SO IT WORKS FOR ALL METHODS
# Other things that may be of interest:
# boxcox: from scipy.stats import boxcox
# Test for stationarity: from statsmodels.tsa.stattools import adfuller
# VARMAX: from statsmodels.tsa.statespace.varmax import VARMAX
# ARIMA: from statsmodels.tsa.arima.model import ARIMA
