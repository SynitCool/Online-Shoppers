import pandas as pd
import numpy as np

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import r_regression

from sklearn.tree import DecisionTreeClassifier


def selection_anova(X, y, n_k_best=10):
    select_best = SelectKBest(f_classif, k=n_k_best)

    selected_features = select_best.fit_transform(X, y)

    return selected_features


def selection_mutual_info(X, y, n_k_best=10):
    select_best = SelectKBest(mutual_info_classif, k=n_k_best)

    selected_features = select_best.fit_transform(X, y)

    return selected_features


def selection_chi2(X, y, n_k_best=10):
    select_best = SelectKBest(chi2, k=n_k_best)

    selected_features = select_best.fit_transform(X, y)

    return selected_features


def selection_variance(X, y, threshold=0):
    variance_threshold = VarianceThreshold(threshold=threshold)

    selected_features = variance_threshold.fit_transform(X, y)

    return selected_features


def selection_decision_tree(X, y, n_k_best=10):
    X_array = np.array(X)
    model = DecisionTreeClassifier()

    model.fit(X, y)

    features_importances = model.feature_importances_
    selected_best_features = sorted(features_importances, reverse=True)[:n_k_best]
    index_highest = np.where(
        np.reshape(selected_best_features, (-1, 1)) == features_importances
    )[1]

    return X_array[:, index_highest]


def selection_pearson_correlation(X, y, thresh_corr=0.1):
    # Init person correlation
    X_array = np.array(X)

    # Calculate pearson correlation
    p_corr = r_regression(X, y)
    thresh_corr = np.array([corr for corr in p_corr if corr >= thresh_corr])

    index_thresh = np.where(np.reshape(thresh_corr, (-1, 1)) == p_corr)[1]

    return X_array[:, index_thresh]
