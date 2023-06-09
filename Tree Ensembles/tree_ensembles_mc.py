import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
from tree_ensembles import run_tree_ensembles

# plt.ion()

n_individuals_range = range(1000, 3001, 1000)


def run_monte_carlo(n_individuals_range, numeric_only_bool):

    with Pool(cpu_count()-1) as pool:
        func = partial(run_tree_ensembles, 5, 10, False, False, numeric_only_bool)
        results = list(pool.imap(func, n_individuals_range))

    return results


def plot_monte_carlo(data: list):

    df_list = []
    for item in data:
        for i, inner_dict in item.items():
            for j, inner_inner_dict in inner_dict.items():
                value = inner_inner_dict['cv_score']
                df_list.append({'i': i, 'Model': j, 'cv_score': value})
    df = pd.DataFrame(df_list)

    num_models = len(df['Model'].unique())
    cmap = plt.get_cmap('Set2')  # Use the Set2 color map

    for i, model in enumerate(df['Model'].unique()):
        model_data = df[df['Model'] == model]
        color = cmap(i % num_models)  # Cycle through the color map
        plt.scatter(model_data['i'], model_data['cv_score'], c=color, label=model, alpha=0.5, s=50)

    plt.xlabel('Number of Individuals')
    plt.ylabel('Cross Validation Scores')
    plt.title('Plot of Cross Validation Scores')
    plt.legend(['Logit', 'Decision Tree', 'Random Forest', 'Adaboost', 'GBM', 'XGBoost'],
               loc='lower right',
               fontsize=9, markerscale=1.5, scatterpoints=1,
               fancybox=True, framealpha=0.5)

    # plt.show()


if __name__ == "__main__":
    mc_output = run_monte_carlo(n_individuals_range, [False])
    plot_monte_carlo(mc_output)
    plt.savefig(f"comparison_false.png", dpi=300)

    mc_output = run_monte_carlo(n_individuals_range, [True])
    plot_monte_carlo(mc_output)
    plt.savefig(f"comparison_true.png", dpi=300)
