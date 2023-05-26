import random
import string
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import xgboost as xgb


class TreeModels:
    def __init__(
            self,
            n_group: int = 5,
            n_individuals: int = 10000,
            n_num_features: int = 5,
    ):
        print(f'There are {n_individuals} people.')
        print(f'There are {n_group} choices.')
        print(f'There are {n_num_features} numerical features and 1 categorical feature.')

        self.num_features = np.random.rand(n_individuals, n_num_features + 2)

        cat_list = random.choices(string.ascii_uppercase, k=5)
        self.cat_feature = np.random.choice(cat_list, size=(n_individuals, 1))

        self.df = pd.DataFrame(self.num_features[:, :-2])
        self.df['cat_feature'] = self.cat_feature
        self.df = pd.get_dummies(self.df, prefix=['cat'])
        self.df.columns = self.df.columns.astype(str)

        kmeans = KMeans(n_clusters=n_group, n_init="auto").fit(self.num_features)
        self.df['target'] = kmeans.labels_

        numerical_columns = [str(i) for i in range(n_num_features)]
        for column in numerical_columns:
            self.df[column] = self.df[column] + random.gauss(mu=0, sigma=3)

        self.X = self.df.drop(columns=['target'])
        self.y = self.df['target']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42)

    def AllResults(
            self,
            clf,
    ):
        clf.fit(self.X_train, self.y_train)
        self.y_pred = clf.predict(self.X_test)
        print(f'Feature Importance: {clf.feature_importances_}')

        train_acc = clf.score(self.X_train, self.y_train)
        print(f'Training accuracy: {train_acc:.4f}')
        acc = accuracy_score(self.y_test, self.y_pred)
        print(f'Test accuracy: {acc:.4f}')
        precision = precision_score(self.y_test, self.y_pred, average='weighted')
        print(f'Test precisions: {precision:.4f}')
        f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        print(f'Test F1 score: {f1:.4f}')

        cv_score = cross_val_score(clf, self.X, self.y, cv=10)
        print(f'Average Cross Validation: {np.mean(cv_score)}')

        cm = confusion_matrix(self.y_test, self.y_pred, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=clf.classes_)
        disp.plot()
        plt.show()


if __name__ == "__main__":
    tree = TreeModels(n_num_features=10)

    d_tree = DecisionTreeClassifier()
    tree.AllResults(d_tree)

    rf = RandomForestClassifier()
    tree.AllResults(rf)

    ada = AdaBoostClassifier()
    tree.AllResults(ada)

    gbm = GradientBoostingClassifier()
    tree.AllResults(gbm)

    xgb = xgb.XGBClassifier()
    tree.AllResults(xgb)
