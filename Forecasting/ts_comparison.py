# plan
# loop over 10 (or more) different stocks
# create a graph where x is N_TEST and y is MAPE
# Models included: ANN, CNN, GRU, LSTM, and Prophet

# Prepare a list of stocks and the value of N_TEST
# get data from `yfinance`
# call ts_tensorflow to run ANN, CNN, GRU, and LSMT and return predictions
# call ts_prophet to run Prophet and return predictions

# ts_tensorflow and ts_prophet both accept a list of stocks and N_TEST
# Need to loop over the list of stocks for the actual `run` methods

from ts_tensorflow import StocksForecastDL
from ts_prophet import StocksForecastProphet


def run_comparison():
    pass


if __name__ == "__main__":
    stock_list = ['AAPL', 'UAL', 'WMT', 'PFE', 'MA', 'MCD', 'OXY', 'BA', 'GE', 'GM']
    N_TEST_list = [5, 10, 20, 50, 100, 200]

    dl = StocksForecastDL(stock_name_list=stock_list)
    p = StocksForecastProphet(stock_name_list=stock_list)

    for stock in stock_list:
        dl.single_model_comparison(stock_name=stock)
