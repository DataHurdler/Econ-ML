import random
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import xgboost

N_GROUP = 5
N_IND = 10000
N_FEATURES = 10


class TreeModels:
    def __init__(
            self,
            n_group: int = 5,
            n_individuals: int = 10000,
            n_num_features: int = 10,
            numeric_only: bool = False,
    ):
        """
        Initialize the TreeModels class.

        Args:
            n_group (int): Number of groups. Default is 5.
            n_individuals (int): Number of individuals. Default is 10000.
            n_num_features (int): Number of numerical features. Default is 10.
            numeric_only (bool): Flag to indicate whether to use only numerical features. Default is False.

        Returns:
            None
        """
        print(f'There are {n_individuals} individuals.')
        print(f'There are {n_group} choices.')
        print(f'There are {n_num_features} numerical features and 1 categorical feature.')

        self.numeric_only = numeric_only

        # Generate random numerical features and categorical feature
        self.num_features = np.random.rand(n_individuals, n_num_features + 2)
        cat_list = random.choices(string.ascii_uppercase, k=6)
        self.cat_features = np.random.choice(cat_list, size=(n_individuals, 1))

        # Create a DataFrame with numerical features and one-hot encoded categorical feature
        self.df = pd.DataFrame(self.num_features[:, :-2])
        self.df['cat_features'] = self.cat_features
        self.df = pd.get_dummies(self.df, prefix=['cat'])
        self.df.columns = self.df.columns.astype(str)

        if numeric_only:
            # Cluster the data based on numerical features only
            # Logistic regression performs the best in this condition
            kmeans = KMeans(n_clusters=n_group, n_init="auto").fit(self.num_features)
            self.df['target'] = kmeans.labels_
        else:
            # Cluster the data based on both numerical and categorical features
            cat_columns = self.df.filter(like='cat')
            kmeans1 = KMeans(n_clusters=n_group, n_init="auto").fit(cat_columns)
            kmeans2 = KMeans(n_clusters=n_group, n_init="auto").fit(self.num_features)
            self.df['target'] = np.floor((kmeans1.labels_ + kmeans2.labels_) / 2)

        # Add some random noise to the numerical features
        numerical_columns = [str(i) for i in range(n_num_features)]
        for column in numerical_columns:
            self.df[column] = self.df[column] + random.gauss(mu=0, sigma=3)

        # Split the data into training and testing sets
        self.X = self.df.drop(columns=['target'])
        self.y = self.df['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42)

        # Initialize the y_pred variable
        self.y_pred = np.empty([n_individuals, 1])

        # Initialize a dictionary to save results
        self.results = dict()

    def show_results(self, clf, clf_name, print_flag=False, save_plot=True):
        """
        Train and evaluate a classifier.

        Args:
            clf: Classifier object.
            clf_name (str): Name of the classifier.
            print_flag (bool): Whether to print results. Default is False.
            save_plot (bool): Whether to save plots. Default is True.

        Returns:
            None
        """
        # print(clf)
        clf.fit(self.X_train, self.y_train)
        self.y_pred = clf.predict(self.X_test)

        # Calculate evaluation metrics
        train_acc = clf.score(self.X_train, self.y_train)
        acc = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred, average='weighted')
        recall = recall_score(self.y_test, self.y_pred, average='weighted')
        f1 = f1_score(self.y_test, self.y_pred, average='weighted')

        # Perform cross-validation and print the average score
        cv_score = cross_val_score(clf, self.X, self.y, cv=10)

        if print_flag:
            if isinstance(clf, LogisticRegression):
                print(f'Coefficients: {clf.coef_}')
            else:
                print(f'Feature Importance: {clf.feature_importances_}')
            print(f'Training accuracy: {train_acc:.4f}')
            print(f'Test accuracy: {acc:.4f}')
            print(f'Test precision: {precision:.4f}')
            print(f'Test recall: {recall:.4f}')
            print(f'Test F1 score: {f1:.4f}')
            print(f'Average Cross Validation: {np.mean(cv_score)}')

        # Plot the confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot()

        if save_plot:
            plt.savefig(f"cm_{clf_name}_{self.numeric_only}.png", dpi=150)

        plt.show()

        # Save results in self.result dictionary
        self.results[clf_name] = {
            'train_acc': train_acc,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_score': np.mean(cv_score)
        }


def run_tree_ensembles(
        n_group: int = 5,
        n_individuals: int = 10000,
        n_num_features: int = 10,
        print_flag: bool = True,
        save_plot: bool = True,
        numeric_only_bool: list = (False, True),
) -> dict:

    for i in numeric_only_bool:
        tree = TreeModels(n_group, n_individuals, n_num_features, numeric_only=i)

        logit = LogisticRegression(max_iter=10000)
        tree.show_results(logit, 'logit', print_flag, save_plot)

        d_tree = DecisionTreeClassifier()
        tree.show_results(d_tree, 'decisiontree', print_flag, save_plot)

        rf = RandomForestClassifier()
        tree.show_results(rf, 'randomforest', print_flag, save_plot)

        ada = AdaBoostClassifier()
        tree.show_results(ada, 'adaboost', print_flag, save_plot)

        gbm = GradientBoostingClassifier()
        tree.show_results(gbm, 'gbm', print_flag, save_plot)

        xgb = xgboost.XGBClassifier()
        tree.show_results(xgb, 'xgboost', print_flag, save_plot)

        return tree.results


if __name__ == "__main__":
    # No interruption by plt.show()
    plt.ion()
    random.seed(123)
    run_tree_ensembles()
