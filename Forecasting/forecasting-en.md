<!-- omit in toc -->
Time Series, Forecasting, and Deep Learning Algorithms
==========

*Zijun Luo*

## Introduction

This chapter is structurally different from other chapters. In the next section, we will first look at Python implementations. We will implement time-series/forecasting models that are not based on machine learning algorithms mainly using the Python library `statsmodels`. This serves both as a review for forecasting concepts and an introduction to the `statsmodels` library, another widely used python library for statistical/data analysis.

We will then cover, briefly, how we can transform a forecasting problem into one of `machine learning`'s. Such transformation allows us to use regression and classification algorithms to tackle complicated forecasting tasks. We have already introduced **classification** algorithms in the last chapter. Machine learning **regression** algorithms, however, is deferred to the next chapter. But readers with a quantitative background should have no problem linking forecasting to regression models.

The main emphasis of this chapter is the use of `deep learning` models for forecasting tasks. For this, we will look at how different neural network models, including `Artificial Neural Networks` (ANN), `Convolutional Neural Networks` (CNN), and `Reccurent Neural Networks` (RNN) may be useful. We will implement some of these methods in Python using `TensorFlow`. Finally, this chapter ends with the introduction to `Facebook`'s `Prophet` library, which is a widely-used library in the industry.

**Forecasting** should require no further introduction. At its simplest form, you have a time series data set, which is contains values of a single object/individual overtime, and you try to predict the "next" value into the future. In more complicated cases, you can have covariates/features, as long as these features are observable at the moment of forecasting and do not result in the **information leakage**"**. For example, if you are doing weather forecast and your goal is to forecast whether it is going to rain tomorrow, then a time-series dataset would contain only information of whether it has rained or not for the past many days, whereas additional features such as temperature, dew point, and precipitation may be included. These additional weather variables should be from the day before your forecast, not the day of your forecast when you are training your model. A class example of information leakage happens when forecasting with moving average (MA) values. For example, if you are doing a 3-day MA, then the value of today requires the use of the value from tomorrow, which is only possible in historic data but not with real data.

## Time Series Implementation in `statsmodels`

In this section, we will implement three forecasting models: `Exponential Smoothing (ETS)`, `Vector Autoregression (VAR)`, and `Auto Autoregressive Integrated Moving Average (ARIMA)`. ETS and ARIMA are run with a single time series, whereas VAR uses several. The data set we will use is U.S. stock exchange (close) prices from the python library `yfinance`. For ETS, we will also implement a walk-forward validation, which is the correct form of validation for time series data, analogue to cross validation seen in the last chapters. To show the powerfulness of Auto Machine Learning, we will implement auto ARIMA from the python library `pmdarima`. Here is the python script:

```python
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

    def walkforward(self, h, steps, tuple_of_option_lists, stock_name='UAL', col='Close'):
        """
        Perform walk-forward validation on the specified stock. Only supports ExponentialSmoothing

        Args:
            h (int): The forecast horizon.
            steps (int): The number of steps to walk forward.
            tuple_of_option_lists (tuple): Tuple of option lists for trend and seasonal types.
            stock_name (str): The name of the stock. Default is 'UAL'.
            col (str): The column name to use for the model. Default is 'Close'.

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
    print(trend_type)
    print(seasonal_type)
```

As in other chapters, a `class`, named `StocksForecast`, is written. In the beginning the script, we have two static methods outside of the class, which is for data preparation and plotting. In side the `StockForecast` class, we initiate the class with:

1. download the data
2. store data into a `dictionary` with each stock in a different key
3. calculate the log and first-differenced values of `close price`.

```python
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
```

Each algorithm is implemented inside a wrapper function. For example, the ETS implementation is inside `run_ets()`, which has only 4 lines:

1. call the `prepare_data()` function
2. instantiate the `ExponentialSmoothing` model from `statsmodels` with hyperparameters `trend`, `seasonal`, and `seasonal_periods`. For `trend` and `seasonal`, `mul` means these trends are multiplicative. The value 252 (days) is used for `seasonal_periods` since this is about the number of trading days in half a year
3. call `model.fit()`
4. call the `plot_fitted_forecast()` model

```python
    def run_ets(self, stock_name='UAL', col='Close'):
        """
        Run the Exponential Smoothing (ETS) model on the specified stock.

        Args:
            stock_name (str): The name of the stock. Default is 'UAL'.
            col (str): The column name to use for the model. Default is 'Close'.
        """
        train, test, train_idx, test_idx = prepare_data(self.dfs[stock_name])

        model = ExponentialSmoothing(train[col].dropna(), trend='mul', seasonal='mul', seasonal_periods=126)
        result = model.fit()

        plot_fitted_forecast(self.dfs[stock_name], train_idx, test_idx, result, col)
```

A walk-forward validation for ETS is implemented in by the method `walkforward()` (from Lazy Programmer). For time series data, we can not perform cross-validation by selecting a random subset of observations, as this can result in using future values to predict past value. Instead, a n-step walk-forward validation should be used. Suppose we have data from 1/1/2018 to 12/31/2022, a 1-step walk-forward validation using data from December 2022 would involve the following steps:

1. train the model with data from 1/1/2018 to 11/30/2022
2. with model result, make prediction for 12/1/2022
3. compare the true and predicted values and calculate the error or other desire metric
4. "walk forward" by 1 day, then go back to training the model, i.e., train the model with data from 1/1/2018 to 12/1/2022
5. continue until data from 1/1/2018 to 12/30/2022 is used for training and 12/31/2022 is predicted

The following method inside the `StocksForecast` class implements the walk-forward validation:

```python
    def walkforward(self, h, steps, tuple_of_option_lists, stock_name='UAL', col='Close'):
        """
        Perform walk-forward validation on the specified stock. Only supports ExponentialSmoothing

        Args:
            h (int): The forecast horizon.
            steps (int): The number of steps to walk forward.
            tuple_of_option_lists (tuple): Tuple of option lists for trend and seasonal types.
            stock_name (str): The name of the stock. Default is 'UAL'.
            col (str): The column name to use for the model. Default is 'Close'.

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

        return np.mean(errors)
```

We should try several different hyperparameter combinations since the purpose of the walk-forward validation is to choose the "best" hyperparameters. The following lines inside `if __name__ == "__main__":` calls the `walkforward()` method to try a combination of hyperparameters and prints out the "best" values for `trend` and `seasonal`:

```python
    H = 20  # 4 weeks
    STEPS = 10

    # Hyperparameters to try
    trend_type_list = ['add', 'mul']
    seasonal_type_list = ['add', 'mul']
    
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
```

The method `run_var()` runs the VAR model. Since we run VAR with several stocks, standardized/normalized should be performed. This is accomplished with the `StandardScaler()` method from scikit-learn after we have called the `prepare_data()` function to split the data into train and test sets. In the `run_var()` method, we also call the methods `plot_acf()` and `plot_pacf()` from sci-kit learn to examine the autocorrelation and partial autocorrelation functions. They are also important for the ARIMA model, but since we will be running Auto ARIMA, we are spared of the task of manually determine the values of AR() and MA().

```python
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
```

Last but not leas, the `run_arima()` method runs the Auto ARIMA from the `pdmarima` library. Similar to `run_ets()`, there are only four lines of code:

```python
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
```

If you would like to run ARIMA from `statsmodels`, you can import `ARIMA` from `statsmodels.tsa.arima.model`. `statsmodels` also provides functions and API for many other time-series/forecasting methods.

## Machine Learning Methods (Brief)

## Self-supervised Learning (?)

## Deep Learning Methods

## Long Short-Term Memory (LSTM)

## Facebook Prophet

## Summary