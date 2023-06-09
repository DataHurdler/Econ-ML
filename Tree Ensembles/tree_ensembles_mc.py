import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
from tree_ensembles import run_tree_ensembles

# plt.ion()

n_individuals_range = range(1000, 3001, 1000)


def run_monte_carlo(n_individuals_range):

    all_results = {}

    # with Pool(cpu_count()-1) as pool:
    #     results = pool.starmap(run_tree_ensembles, [
    #         (5, n_individuals, 10, False, False, [True]) for n_individuals in n_individuals_range
    #     ])

    with Pool(cpu_count()-1) as pool:
        func = partial(run_tree_ensembles, 5, 10, False, False, [True])
        results = list(pool.imap(func, n_individuals_range))

        # all_results[n_individuals] = results

    return results


def plot_monte_carlo(data: list):

    df_list = []
    for item in data:
        for i, inner_dict in item.items():
            for j, inner_inner_dict in inner_dict.items():
                value = inner_inner_dict['cv_score']
                df_list.append({'i': i, 'Model': j, 'cv_score': value})
    df = pd.DataFrame(df_list)

    colors = sns.color_palette("Set2", len(data))
    
    sns.scatterplot(data=df, x='i', y='cv_score', hue='Model', legend="full", palette=colors)
    plt.xlabel('Number of Individuals')
    plt.ylabel('Cross Validation Scores')
    plt.title('Plot of Cross Validation Scores')
    plt.legend(['Logit', 'Decision Tree', 'Random Forest', 'Adaboost', 'GBM', 'XGBoost'],
               loc='lower right',
               fontsize=9, markerscale=1.5, scatterpoints=1,
               fancybox=True, framealpha=0.5)
    plt.show()


if __name__ == "__main__":
    mc_output = run_monte_carlo(n_individuals_range)

    # for n_individuals in n_individuals_range:
    #     print(mc_output[n_individuals])
    #
    # for key in mc_output:
    #     print(mc_output[key])

    # print(mc_output)
    plot_monte_carlo(mc_output)
