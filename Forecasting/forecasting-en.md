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


def plot_fitted_forecast(df, col=None):
    """
    Plot the fitted and forecasted values of a time series.

    Args:
        df (pandas.DataFrame): The input dataframe.
        col (str): The column name to plot. Default is None.
    """
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
        df_all = self.dfs[stock_name]
        train, test, train_idx, test_idx = prepare_data(df_all)

        model = ExponentialSmoothing(train[col].dropna(), trend='mul', seasonal='mul', seasonal_periods=252)
        result = model.fit()

        df_all.loc[train_idx, 'fitted'] = result.fittedvalues
        df_all.loc[test_idx, 'forecast'] = np.array(result.forecast(N_TEST))

        plot_fitted_forecast(df_all, col)

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

        # standardizing different stocks
        for value in stock_cols:
            scaler = StandardScaler()
            train[f'Scaled_{value}'] = scaler.fit_transform(train[[value]])
            test[f'Scaled_{value}'] = scaler.transform(test[[value]])
            df_all.loc[train_idx, f'Scaled_{value}'] = train[f'Scaled_{value}']
            df_all.loc[test_idx, f'Scaled_{value}'] = test[f'Scaled_{value}']

        cols = ['Scaled_' + value for value in stock_cols]

        return df_all, train, test, train_idx, test_idx, stock_cols, cols

    def run_var(self, stock_list=('UAL', 'WMT', 'PFE'), col='Close'):
        """
        Run the Vector Autoregression (VAR) model on the specified stocks.

        Args:
            stock_list (tuple): Tuple of stock names. Default is ('UAL', 'WMT', 'PFE').
            col (str): The column name to use for the model. Default is 'Close'.
        """

        df_all, train, test, train_idx, test_idx, stock_cols, cols = self.prepare_data_var(stock_list, col)

        model = VAR(train[cols])
        result = model.fit(maxlags=40, method='mle', ic='aic')

        lag_order = result.k_ar
        prior = train.iloc[-lag_order:][cols].to_numpy()
        forecast_df = pd.DataFrame(result.forecast(prior, N_TEST), columns=cols)

        df_all.loc[train_idx, 'fitted'] = result.fittedvalues[cols[0]]
        df_all.loc[test_idx, 'forecast'] = forecast_df[cols[0]].values

        col = "Scaled_" + stock_cols[0]
        plot_fitted_forecast(df_all, col)

        # Calculate R2
        print("VAR Train R2: ", r2_score(df_all.loc[train_idx, cols[0]].iloc[lag_order:],
                                         df_all.loc[train_idx, 'fitted'].iloc[lag_order:]))
        print("VAR Test R2: ", r2_score(df_all.loc[test_idx, cols[0]],
                                        df_all.loc[test_idx, 'forecast']))

    def run_arima(self, stock_name='UAL', col='Close', seasonal=True, m=12):
        """
        Run the Auto Autoregressive Integrated Moving Average (ARIMA) model on the specified stock.

        Args:
            stock_name (str): The name of the stock. Default is 'UAL'.
            col (str): The column name to use for the model. Default is 'Close'.
            seasonal (bool): Whether to include seasonal components. Default is True.
            m (int): The number of periods in each seasonal cycle. Default is 12.
        """
        df_all = self.dfs[stock_name]
        train, test, train_idx, test_idx = prepare_data(df_all)

        plot_acf(train[col])
        plot_pacf(train[col])

        model = pm.auto_arima(train[col], trace=True, suppress_warnings=True, seasonal=seasonal, m=m)

        print(model.summary())

        df_all.loc[train_idx, 'fitted'] = model.predict_in_sample(end=-1)
        df_all.loc[test_idx, 'forecast'] = np.array(model.predict(n_periods=N_TEST, return_conf_int=False))

        plot_fitted_forecast(df_all, col)


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

    tuple_of_option_lists = (trend_type_list, seasonal_type_list,)
    ts.run_walkforward(H, STEPS, STOCK, COL, tuple_of_option_lists)
```

As in other chapters, a `class`, named `StocksForecast`, is written. In the beginning the script, we have two static methods/functions outside of the class for data preparation and plotting. In side the `StockForecast` class, we initiate the class with:

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

Each algorithm is implemented inside a wrapper function. For example, the ETS implementation is in `run_ets()`, which does the following:

1. call the `prepare_data()` function
2. instantiate the `ExponentialSmoothing` model from `statsmodels` with hyperparameters `trend`, `seasonal`, and `seasonal_periods`. For `trend` and `seasonal`, `mul` means these trends are multiplicative. The value 252 (days) is used for `seasonal_periods` since this is about the number of trading days in half a year
3. call `model.fit()`
4. get forecast columns and prepare the data for plotting
5. call the `plot_fitted_forecast()` function to plot

```python
    def run_ets(self, stock_name='UAL', col='Close'):
        """
        Run the Exponential Smoothing (ETS) model on the specified stock.

        Args:
            stock_name (str): The name of the stock. Default is 'UAL'.
            col (str): The column name to use for the model. Default is 'Close'.
        """
        df_all = self.dfs[stock_name]
        train, test, train_idx, test_idx = prepare_data(df_all)

        model = ExponentialSmoothing(train[col].dropna(), trend='mul', seasonal='mul', seasonal_periods=252)
        result = model.fit()

        df_all.loc[train_idx, 'fitted'] = result.fittedvalues
        df_all.loc[test_idx, 'forecast'] = np.array(result.forecast(N_TEST))

        plot_fitted_forecast(df_all, col)
```

A walk-forward validation for ETS is implemented in by the method `run_walkforward()` (from Lazy Programmer) which is a wrapper function of `warlkforward_ets()`. For time series data, we can not perform cross-validation by selecting a random subset of observations, as this can result in using future values to predict past value. Instead, a n-step walk-forward validation should be used. Suppose we have data from 1/1/2018 to 12/31/2022, a 1-step walk-forward validation using data from December 2022 would involve the following steps:

1. train the model with data from 1/1/2018 to 11/30/2022
2. with model result, make prediction for 12/1/2022
3. compare the true and predicted values and calculate the error or other desire metric
4. "walk forward" by 1 day, then go back to training the model, i.e., train the model with data from 1/1/2018 to 12/1/2022
5. continue until data from 1/1/2018 to 12/30/2022 is used for training and 12/31/2022 is predicted

The following method inside the `StocksForecast` class implements the walk-forward validation:

We should try several different hyperparameter combinations since the purpose of the walk-forward validation is to choose the "best" hyperparameters. The following lines inside `if __name__ == "__main__":` calls the `run_walkforward()` method to try a combination of hyperparameters and prints out the "best" values for `trend` and `seasonal`:

```python
    H = 20  # 4 weeks
    STEPS = 10

    # Hyperparameters to try in ETS walk-forward validation
    trend_type_list = ['add', 'mul']
    seasonal_type_list = ['add', 'mul']

    tuple_of_option_lists = (trend_type_list, seasonal_type_list,)
    ts.run_walkforward(H, STEPS, STOCK, COL, tuple_of_option_lists)
```

The method `run_var()` runs the VAR model. Since we run VAR with several stocks, standardized/normalized should be performed. This is accomplished in the `prepare_data_var()` method with `StandardScaler()` from scikit-learn. 

Last but not leas, the `run_arima()` method runs the Auto ARIMA from the `pdmarima` library. Here, we also call `plot_acf()` and `plot_pacf()` from sci-kit learn to examine the autocorrelation and partial autocorrelation functions. Normally, they are important for the ARIMA model. However, with Auto ARIMA, we are spared of the task of manually determine the values of AR() and MA(). Similar to `run_ets()`, there are only a few lines of code:

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
        df_all = self.dfs[stock_name]
        train, test, train_idx, test_idx = prepare_data(df_all)

        plot_acf(train[col])
        plot_pacf(train[col])

        model = pm.auto_arima(train[col], trace=True, suppress_warnings=True, seasonal=seasonal, m=m)

        print(model.summary())

        df_all.loc[train_idx, 'fitted'] = model.predict_in_sample(end=-1)
        df_all.loc[test_idx, 'forecast'] = np.array(model.predict(n_periods=N_TEST, return_conf_int=False))

        plot_fitted_forecast(df_all, col)
```

If you would like to run ARIMA from `statsmodels`, you can import `ARIMA` from `statsmodels.tsa.arima.model`. `statsmodels` also provides functions and APIs for other time-series/forecasting methods and models. For example, you can test fo stationarity with the augmented Dickey-Fuller unit root test by importing `adfuller` from `statsmodels.tsa.stattools`, or run the Vector Autoregressive Moving Average with exogenous regressors by importing `VARMAX` from `statsmodels.tsa.statespace.varmax`. In addition, if you would like to do the Box-Cox transformation, you can import `boxcox` from `scipy.stats`.

## Machine Learning Methods

Briefly mentions Self-supervised Learning?

## Artificial Neural Network (ANN)

Similar to other chapters, the assumption is that readers have some idea about what a neural network is and what it can do. Our goal is not to give an in-depth introduction to neural networks. Rather, we will only cover elements of neural networks that matter most in their applications in economics and business assuming readers already have some quantitative training. An excellent place that you can "play" with a neural network model is the [Tensorflow Playground](https://playground.tensorflow.org/).

Neural networks can be used on both regression and classification problems. Our focus in this chapter is to use neural networks on regression since the emphasis is forecasting. Keep in mind that we can always reshape a regression problem into a classification problem. For example, instead of forecasting the actual price or return of a stock, we can predict the likelihood of a stock trending up or down, which is a binary classification problem. The difference between applying neural networks on regression or classification problems is minor: for regression problems, the final activation function is an identify function (returns itself) whereas for classification problems it is Sigmoid or other functions that return values between 0 and 1. A really good summary of activation functions is [this answer on stackexchange](https://stats.stackexchange.com/questions/115258/comprehensive-list-of-activation-functions-in-neural-networks-with-pros-cons).

Let us begin with artificial neural network (ANN). For implementation of neural networks, we are using `Keras` (https://keras.io/) from `Tensorflow` (https://www.tensorflow.org/). We will introduce `PyTorch`, another popular deep learning framework, in other chapters.

Neural network models intend to mimic the human brain. The basic idea can be described as follow. Imagine you see an ice-cream truck and decide to try an ice-cream that you have not had before. First, you receive multiple signals: you see the brand, shape, color, and possibly smell and ingredients of many ice-creams that you can choose from. These "raw" signals are first passed through the initial layer of neurons, the ones immediately connected to your eyes and noses and other sensory organs. After the initial layer and processing, you recognize different features of many ice-creams, some excites you, some not. In neural science terminology, the outputs from the first layer of neurons have different "action potential". If the action potential passes a certain threshold, it excites you. But such excitement can be both positive and negative. For example, you may recognize there are peanuts in some of the ice-creams cones. While the crunchy cone excites you, you also know that you are allergic to peanuts. Imagine in the second layer, one neuron specializes in recognizing cones and the other peanuts. The output from the first layer would activate both of these two neurons. And hence the name "activation function". This process can continue. A neural network may contain many layers, and each layer many neurons. After passing through all the layers, you have arrived at your decision: A cup with vanilla and strawberry ice-creams and chocolate chips on top.

Suppose your raw data set has $N$ observations/rows and $M$ features/columns. The probability of the $i$'s neuron in the first layer being activated is

$$z^{(1)}_i=p(\text{activated} \mid x)=\sigma(xW^{(1)}_i+b^{(1)}_i)$$

where $x$ is a $N\times M$ matrix, $W^{(1)}_i$ and $b^{(1)}_i$ are both vectors of size $M$, and $\sigma()$ is an activation function that returns a probability such as Sigmoid or ReLU. In regression terminology, $W^{(1)}_i$ are the coefficients and $b^{(1)}_i$ is the intercept. By neural network convention, we use the superscript $(j)$ to denote layer.

Usually each layer has multiple neurons. In this case, the outputs $z^{(j)}_i$ can be "stacked" horizontally and fed into the next lay. We an similarly stack $W^{(j)}_i$ and $b^{(j)}_i$. In other words, the number of neurons in the current layer ($j$) is the number of features for the next layer layer ($j+1$). With this, we can express the whole neural network in the following manner:

- Beginning ($j=1$): $z^{(1)}=\sigma(xW^{(1)}+b^{(1)})$
- Hidden layers ($1<j<J$): $z^{(j)}=\sigma(z^{(j-1)}W^{(j)}+b^{(j)})$
- Final layer ($j=J$): $\hat{y}=z^{(L-1)}W^{(L)}+b^{(L)}$

where $J$ denotes the total number of layers, and $\hat{y}$ is the prediction. Note that the final layer does not have an activation function here because we are dealing with a regression model.

While Sigmoid is a widely used function when probabilities are to be predicted, it suffers from the [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) especially with deep (many layers) neural networks. Modern deep learning models often use `ReLU` or `tanh` as the activation function for inner layers. Again, see [this answer on stackexchange](https://stats.stackexchange.com/questions/115258/comprehensive-list-of-activation-functions-in-neural-networks-with-pros-cons) for the pros and cons of different activation functions in neural networks.

### ANN in code

## Recurrent Neural Network (RNN)

There is a compelling reason why `Recurrent Neural Network` (RNN) models are often expected to perform well in time-series/forecasting tasks: it is the neural network version of the autoregressive (AR) process. in its simplest form, often referred to as `Simple RNN`, the output from the hidden layers of time $t-1$ is used as inputs for time $t$ in addition to $x$.

Suppose you only care about one-step forecast, i.e., you want to predict $t+1$ with data up to time $t$. Suppose we use all data for training, the approaches covered in this chapter so far have basically the same flavor: specify a single model for any length of time, train the model using data up to time $t$, and make the prediction for $t+1$. Even with walk-forward validation, it is not much different except that several values of $t$ are considered and hence the model was trained on different data and can have different parameters dependent on the value of $t$.

Having a single unified model is often fine as long as the time series does not have large ups and downs. Unfortunately, economics and business time-series data only consists of ups and downs, such as a recession. In such cases, we often want to specify more than one model. That can be accomplished manually if we know exactly when a structural break has happened.

But life is a box of chocolates and every hour/day is different. It would be nice that a model can do the following: that it "remembers" the past and customizes a model for the current time.

RNN does exactly that. Concretely, let $h_t$ denote the *hidden state* of an RNN at time $t$, we have

$$h_t = \sigma(h_{t-1}W_{ht} + x_tW_{xt}+b_{t})$$

where $W_{ht}$ and $W_{xt}$ are coefficients/weights for the hidden state and input $x_t$, respectively, at time $t$, and $b_{t}(= b_{ht} + b_{xt})$ is the intercept. The hidden state allows the model to "remember" the past and adds non-linear complexity to each time period. It should be noted that $h_t$ can be a mini ANN with many hidden layers.

In addition to Simple RNN, `Long Short-Term Member` (LSTM) and `Gated Recurrent Units` (GRU) are two widely used RNN models. Both models modified how hidden state is being remembered from one time period (or one state) to another. For GRU, two "gates" are introduced:

- Update gate: $z_t = \sigma(x_tW_{xzt}+h_{t-1}W_{hzt}+b_{zt})$
- Reset gate: $r_t = \sigma(x_tW_{xrt}+h_{t-1}W_{hrt}+b_{rt})$

And the hidden state is updated according to

$$h_t = (1-z_t)\odot h_{t-1} + z_t\odot \omega(x_tW_{xht}+(r_t\odot h_{t-1})W_{hht}+b_{ht})$$

where $\odot$ is an element-wise multiplication and $\omega()$ is an activation function similar to $\sigma()$ except that in Tensorflow the default is `tanh` instead of Sigmoid for RNN. In the GRU, $z_t$ controls how much the neural network "forgets" and $r_t$ controls how much the neural network "learns" from the previous state. If $z_t=0$, then the neural network forgets about the previous state (since $1-z_t=0$) and relearn. Keep in mind that the relearn, which is $\omega()$ still consists of the previous hidden state $h_{t-1}$ unless $r_t$ is also equal to 0.

For LSTM, we introduce a new state called `cell state` in addition to the hidden state. In practice, the cell state is an intermediate value that helps to keep track of the model is not included in calculating the final output. The LSTM has three gates:

- Forget gate: $f_t = \sigma(x_tW_{xft}+h_{t-1}W_{hft}+b_{ft})$
- Input/Update gate: $i_t = \sigma(x_tW_{xit}+h_{t-1}W_{hit}+b_{it})$
- Output gate: $o_t = \sigma(x_tW_{xot}+h_{t-1}W_{hot}+b_{ot})$

And the hidden state and cell state ($c_t$) are updated according to:

- Cell state: $c_t = f_t\odot c_{t-1} + i_t\odot \omega(x_tW_{xct}+h_{t-1}W_{hct}+b_{ct})$
- Hidden state: $h_t = o_t\odot \psi(c_t)$

Note that in Tensorflow, the activation function $\omega()$ and $\psi()$ can not be specified individually and are both defaulted to tanh.

### RNN in code

## Convolutional Neural Network (CNN)

https://towardsdatascience.com/fourier-transform-for-time-series-292eb887b101

## Facebook Prophet

https://facebook.github.io/prophet/

## Summary

## References

https://www.udemy.com/course/time-series-analysis/