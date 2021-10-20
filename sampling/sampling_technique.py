import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import RandomOverSampler

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids


def over_sampling_smote(X, y):
    sampler = SMOTE(random_state=42)

    resample_X, resample_y = sampler.fit_resample(X, y)

    return resample_X, resample_y


def over_sampling_adasyn(X, y):
    sampler = ADASYN(random_state=42)

    resample_X, resample_y = sampler.fit_resample(X, y)

    return resample_X, resample_y


def over_sampling_svmsmote(X, y):
    sampler = SVMSMOTE(random_state=42)

    resample_X, resample_y = sampler.fit_resample(X, y)

    return resample_X, resample_y


def over_sampling_random(X, y):
    sampler = RandomOverSampler(random_state=42)

    resample_X, resample_y = sampler.fit_resample(X, y)

    return resample_X, resample_y


def under_sampling_random(X, y):
    sampler = RandomUnderSampler(random_state=42)

    resample_X, resample_y = sampler.fit_resample(X, y)

    return resample_X, resample_y


def under_sampling_cluster_centroids(X, y):
    sampler = ClusterCentroids(random_state=42)

    resample_X, resample_y = sampler.fit_resample(X, y)

    return resample_X, resample_y
