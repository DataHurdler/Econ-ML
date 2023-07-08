import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.plot import plot_cross_validation_metric
from prophet.plot import add_changepoints_to_plot


class StocksForecastProphet:
    def __init__(self,
                 stock_name_list=('UAL', 'WMT', 'PFE'),
                 start_date='2018-01-01',
                 end_date='2022-12-31',
                 ):
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
            self.dfs[name]['DiffLog'] = self.dfs[name]['Log'].diff(1)

            # rename according to prophet rule
            self.dfs[name]['ds'] = self.dfs[name].index

    def run_prophet(self,
                    stock_name: str = 'UAL',
                    col: str = 'Log',
                    diff: bool = True,
                    ):

        df = self.dfs[stock_name]
        df['y'] = df[col]
        print(df.head())

        m = Prophet(yearly_seasonality=True,
                    weekly_seasonality=True)
        m.fit(df)

        # Predict one year ahead
        future = m.make_future_dataframe(periods=365)
        future.tail()

        forecast = m.predict(future)

        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

        fig1 = m.plot(forecast, figsize=(28, 8))

        fig2 = m.plot_components(forecast)

        # A different way of plotting

        plot_plotly(m, forecast)
        plot_components_plotly(m, forecast)

        df_cv = cross_validation(
            m,
            initial='730 days',
            period='30 days',
            horizon='30 days'
        )

        print(df_cv)

        pm = performance_metrics(df_cv)
        print(pm)

        plot_cross_validation_metric(df_cv, metric='rmse')

        plot_cross_validation_metric(df_cv, metric='mape')

        fig = m.plot(forecast, figsize=(28, 8))
        a = add_changepoints_to_plot(fig.gca(), m, forecast)


if __name__ == "__main__":

    ts = StocksForecastProphet()
    ts.run_prophet()
