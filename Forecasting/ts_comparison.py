import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_percentage_error

from ts_tensorflow import StocksForecastDL
from ts_prophet import StocksForecastProphet


def run_comparison(col, stock_list, N_TEST_list, DL_models, epochs, save_df=False):
    """
    Run a comparison of different forecasting models.

    Args:
        col (str): Column name to forecast.
        stock_list (tuple): List of stock names.
        N_TEST_list (tuple): List of forecast horizons.
        DL_models (dict): Dictionary of DL models to compare.
        epochs (int): Number of training epochs.
        save_df (bool, optional): Whether to save the results as a CSV file. Default is False.

    Returns:
        pandas.DataFrame: DataFrame containing the comparison results.
    """

    column_names = ['model', 'stock', 'n_test', 'mape']
    results = pd.DataFrame(columns=column_names)

    for n in N_TEST_list:
        for stock in stock_list:
            p = StocksForecastProphet(stock_name_list=stock_list, n_test=n)

            p.run_prophet(col=col, stock_name=stock, plot=False)
            mape = mean_absolute_percentage_error(p.dfs[stock]['yhat'][-n:], p.dfs[stock]['y'][-n:])
            new_row = ['prophet', stock, n, mape]
            results.loc[len(results)] = new_row
            for model in DL_models:
                dl = StocksForecastDL(stock_name_list=stock_list, n_test=n, epochs=epochs)
                if DL_models[model][0] == 'rnn':
                    dl.single_model_comparison(col=[col],
                                               stock_name=stock,
                                               model=DL_models[model][0],
                                               rnn_model=DL_models[model][1],
                                               plot=False,)
                else:
                    dl.single_model_comparison(col=[col],
                                               stock_name=stock,
                                               model=DL_models[model][0],
                                               plot=False,)

                mape1 = mean_absolute_percentage_error(dl.dfs[stock]['multistep'][-n:], dl.dfs[stock][col][-n:])
                mape2 = mean_absolute_percentage_error(dl.dfs[stock]['multioutput'][-n:], dl.dfs[stock][col][-n:])
                new_row1 = [f"{model}_single", stock, n, mape1]
                new_row2 = [f"{model}_multi", stock, n, mape2]
                results.loc[len(results)] = new_row1
                results.loc[len(results)] = new_row2

    if save_df:
        results.to_csv('results.csv', index=False)

    return results


def plot_comparison(df, load_df=False):
    """
    Plot the comparison results.

    Args:
        df (pandas.DataFrame): DataFrame containing the comparison results.
        load_df (bool, optional): Whether to load the results from a CSV file. Default is False.
    """

    if load_df:
        df = pd.read_csv("results.csv")

    # Calculate the average mape for each model and type
    averages = df.groupby(['model', 'n_test'])['mape'].mean().reset_index()

    # Create a line plot for each model
    fig, ax = plt.subplots(figsize=(15, 5))
    models = averages['model'].unique()
    for model in models:
        # Determine the line style based on the model name
        linestyle = '--o' if 'single' in model else '-o'

        model_data = averages[averages['model'] == model]
        ax.plot(model_data['n_test'], model_data['mape'], linestyle, label=model)

    # Set plot title and labels
    ax.set_title('Average Value by Type for Each Model')
    ax.set_xlabel('Forecast Horizon')
    ax.set_ylabel('Mean Absolute Percentage Error')

    # Show a legend indicating the models
    plt.legend()

    # Save the plot
    plt.savefig("comparison_dl.png", dpi=300)


if __name__ == "__main__":
    plt.ion()

    COL = 'Log'
    # N_TEST_list = (5, 10)
    # stock_list = ('AAPL', 'UAL')
    STOCK_LIST = ('AAPL', 'UAL', 'WMT', 'PFE', 'MA', 'MCD', 'OXY', 'BA', 'GE', 'GM')
    N_TEST_LIST = (5, 10, 20, 50, 100, 200)
    DL_MODELS = {
        "ann": ('ann', None),
        "cnn": ('cnn', None),
        "lstm": ('rnn', 'lstm'),
        "gru": ('rnn', 'gru'),
    }
    EPOCHS = 1500

    comp_results = run_comparison(COL, STOCK_LIST, N_TEST_LIST, DL_MODELS, EPOCHS)
    plot_comparison(comp_results)

    # column_names = ['model', 'stock', 'n_test', 'mape']
    # results = pd.DataFrame(columns=column_names)
    #
    # for n in N_TEST_list:
    #     for stock in stock_list:
    #         p = StocksForecastProphet(stock_name_list=stock_list, n_test=n)
    #
    #         p.run_prophet(col=col, stock_name=stock)
    #         mape = mean_absolute_percentage_error(p.dfs[stock]['yhat'][-n:], p.dfs[stock]['y'][-n:])
    #         new_row = ['prophet', stock, n, mape]
    #         results.loc[len(results)] = new_row
    #         for model in DL_models:
    #             dl = StocksForecastDL(stock_name_list=stock_list, n_test=n, epochs=50)
    #             if DL_models[model][0] == 'rnn':
    #                 dl.single_model_comparison(col=[col],
    #                                            stock_name=stock,
    #                                            model=DL_models[model][0],
    #                                            rnn_model=DL_models[model][1])
    #             else:
    #                 dl.single_model_comparison(col=[col],
    #                                            stock_name=stock,
    #                                            model=DL_models[model][0])
    #
    #             mape1 = mean_absolute_percentage_error(dl.dfs[stock]['multistep'][-n:], dl.dfs[stock][col][-n:])
    #             mape2 = mean_absolute_percentage_error(dl.dfs[stock]['multioutput'][-n:], dl.dfs[stock][col][-n:])
    #             new_row1 = [f"{model}_single", stock, n, mape1]
    #             new_row2 = [f"{model}_multi", stock, n, mape2]
    #             results.loc[len(results)] = new_row1
    #             results.loc[len(results)] = new_row2
    #
    # print(results.head(10))
    # results.to_csv('results.csv', index=False)

    # df = pd.read_csv("results.csv")
    #
    # # Calculate the average mape for each model and type
    # averages = df.groupby(['model', 'n_test'])['mape'].mean().reset_index()
    #
    # # Create a line plot for each model
    # fig, ax = plt.subplots(figsize=(15, 5))
    # models = averages['model'].unique()
    # for model in models:
    #     # Determine the line style based on the model name
    #     linestyle = '--o' if 'single' in model else '-o'
    #
    #     model_data = averages[averages['model'] == model]
    #     ax.plot(model_data['n_test'], model_data['mape'], linestyle, label=model)
    #
    # # Set plot title and labels
    # ax.set_title('Average Value by Type for Each Model')
    # ax.set_xlabel('Forecast Horizon')
    # ax.set_ylabel('Mean Absolute Percentage Error')
    #
    # # Show a legend indicating the models
    # plt.legend()
    #
    # # Save the plot
    # plt.savefig("comparison_dl.png", dpi=300)