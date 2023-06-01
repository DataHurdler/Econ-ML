import random
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb

N_GROUP = 5
N_IND = 10000
N_FEATURES = 10


class TreeModels:
    def __init__(
            self,
            n_group: int = 5,
            n_individuals: int = 10000,
            n_num_features: int = 5,
            numeric_only: bool = False,
    ):
        """
        Initialize the TreeModels class.

        Args:
            n_group (int): Number of groups. Default is 5.
            n_individuals (int): Number of individuals. Default is 10000.
            n_num_features (int): Number of numerical features. Default is 5.
            numeric_only (bool): Flag to indicate whether to use only numerical features. Default is False.

        Returns:
            None
        """
        print(f'There are {n_individuals} individuals.')
        print(f'There are {n_group} choices.')
        print(f'There are {n_num_features} numerical features and 1 categorical feature.')

        # Generate random numerical features and categorical feature
        self.num_features = np.random.rand(n_individuals, n_num_features + 2)
        cat_list = random.choices(string.ascii_uppercase, k=6)
        self.cat_feature = np.random.choice(cat_list, size=(n_individuals, 1))

        # Create a DataFrame with numerical features and one-hot encoded categorical feature
        self.df = pd.DataFrame(self.num_features[:, :-2])
        self.df['cat_feature'] = self.cat_feature
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

    def show_results(
            self,
            clf,
    ):
        """
        Train and evaluate a classifier.

        Args:
            clf: Classifier object.

        Returns:
            None
        """
        print(clf)
        clf.fit(self.X_train, self.y_train)
        self.y_pred = clf.predict(self.X_test)

        if clf == logit:
            print(f'Coefficients: {clf.coef_}')
        else:
            print(f'Feature Importance: {clf.feature_importances_}')

        # Calculate evaluation metrics
        train_acc = clf.score(self.X_train, self.y_train)
        print(f'Training accuracy: {train_acc:.4f}')
        acc = accuracy_score(self.y_test, self.y_pred)
        print(f'Test accuracy: {acc:.4f}')
        precision = precision_score(self.y_test, self.y_pred, average='weighted')
        print(f'Test precision: {precision:.4f}')
        recall = recall_score(self.y_test, self.y_pred, average='weighted')
        print(f'Test recall: {recall:.4f}')
        f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        print(f'Test F1 score: {f1:.4f}')

        # Perform cross-validation and print the average score
        cv_score = cross_val_score(clf, self.X, self.y, cv=10)
        print(f'Average Cross Validation: {np.mean(cv_score)}')

        # Plot the confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=clf.classes_)
        disp.plot()
        plt.show()


if __name__ == "__main__":
    # No interruption by plt.show()
    plt.ion()
    random.seed(123)

    tree = TreeModels(n_num_features=N_FEATURES, numeric_only=False)

    logit = LogisticRegression(max_iter=10000)
    tree.show_results(logit)

    d_tree = DecisionTreeClassifier()
    tree.show_results(d_tree)

    rf = RandomForestClassifier()
    tree.show_results(rf)

    ada = AdaBoostClassifier()
    tree.show_results(ada)

    gbm = GradientBoostingClassifier()
    tree.show_results(gbm)

    xgb = xgb.XGBClassifier()
    tree.show_results(xgb)
