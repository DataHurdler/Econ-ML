import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from prophet.plot import add_changepoints_to_plot


class StocksForecastProphet:
    def __init__(
        self,
        stock_name_list=('UAL', 'WMT', 'PFE'),
        start_date='2018-01-01',
        end_date='2022-12-31',
        n_test=12,
    ):
        """
        Initialize the StocksForecastProphet class.

        Args:
            stock_name_list (tuple): List of stock names.
            start_date (str): Start date for data retrieval.
            end_date (str): End date for data retrieval.
            n_test (int): Number of test samples.
        """
        self.N_TEST = n_test
        self.dfs = dict()
        for name in stock_name_list:
            self.dfs[name] = yf.download(name, start=start_date, end=end_date)
            self.dfs[name]['Diff'] = self.dfs[name]['Close'].diff(1)
            self.dfs[name]['Log'] = np.log(self.dfs[name]['Close'])
            self.dfs[name]['DiffLog'] = self.dfs[name]['Log'].diff(1)
            self.dfs[name]['ds'] = self.dfs[name].index
            self.dfs[f"f_{name}"] = pd.DataFrame()

    def run_prophet(
        self,
        stock_name: str = 'UAL',
        col: str = 'Log',
        cv = False,
        diff: bool = True,
    ):
        """
        Run the Prophet forecasting model for a given stock.

        Args:
            stock_name (str): Name of the stock.
            col (str): Column to be used for forecasting.
            diff (bool): Indicates whether differencing is applied.
        """
        df0 = self.dfs[stock_name]
        df0['y'] = df0[col]
        df = df0[:-self.N_TEST].copy()

        m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        m.fit(df)

        future = m.make_future_dataframe(periods=self.N_TEST)
        self.dfs[f"f_{stock_name}"] = m.predict(future)
        forecast = m.predict(future)

        if cv:
            df_cv = cross_validation(
                m,
                initial='730 days',
                period='30 days',
                horizon='30 days',
                disable_tqdm=True,
            )

            pm = performance_metrics(df_cv)
            print(pm)

            plot_cross_validation_metric(df_cv, metric='mape')

        fig = m.plot(forecast, figsize=(28, 8))
        a = add_changepoints_to_plot(fig.gca(), m, forecast)
        fig2 = m.plot_components(forecast)


if __name__ == "__main__":
    ts = StocksForecastProphet()
    ts.run_prophet()

# https://www.google.com/search?client=firefox-b-1-d&q=mac+how+to+open+usr%2Flocal+in+finder
