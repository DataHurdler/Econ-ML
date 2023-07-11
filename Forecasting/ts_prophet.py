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
            n_test (int): Length of forecast horizon.
        """
        self.N_TEST = n_test
        self.dfs = dict()
        for name in stock_name_list:
            self.dfs[name] = yf.download(name, start=start_date, end=end_date)
            self.dfs[name]['Diff'] = self.dfs[name]['Close'].diff(1)
            self.dfs[name]['Log'] = np.log(self.dfs[name]['Close'])
            self.dfs[name]['DiffLog'] = self.dfs[name]['Log'].diff(1)
            self.dfs[name]['ds'] = self.dfs[name].index

    def run_prophet(
        self,
        stock_name: str = 'UAL',
        col: str = 'Log',
        cv: bool = False,
        plot: bool = True,
    ):
        """
        Run the Prophet forecasting model for a given stock.

        Args:
            stock_name (str): Name of the stock.
            col (str): Column to be used for forecasting.
            cv (bool): Whether to conduct cross validation. Default is False.
            plot (bool): Whether to plot. Default is True.
        """
        df0 = self.dfs[stock_name]
        df0['y'] = df0[col]
        df = df0[:-self.N_TEST].copy()

        m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        m.fit(df)

        future = self.dfs[stock_name]['ds'].reset_index()
        forecast = m.predict(future)
        self.dfs[stock_name]['yhat'] = forecast['yhat'].values

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
            plt.savefig("prophet_cv_mape.png", dpi=300)

        if plot:
            fig = m.plot(forecast, figsize=(28, 8))
            add_changepoints_to_plot(fig.gca(), m, forecast)
            plt.savefig("prophet_with_changepoints.png", dpi=300)

            m.plot_components(forecast)
            plt.savefig("prophet_components", dpi=300)


if __name__ == "__main__":
    ts = StocksForecastProphet()
    ts.run_prophet(cv=True)

# Making Prophet to work with Python 3.9 in MacBook Pro:
# https://www.google.com/search?client=firefox-b-1-d&q=mac+how+to+open+usr%2Flocal+in+finder
