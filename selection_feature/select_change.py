import numpy as np

from sklearn.feature_selection import r_regression
from sklearn.preprocessing import OneHotEncoder


def sc_onehot_pearson_correlation(X, y, thresh_corr=0.1):
    # Initialize
    X = np.array(X)

    # Cal pearson correlation
    p_corr = r_regression(X, y)

    # Find greater than thresh
    thresh_greater_corr = np.array([corr for corr in p_corr if corr >= thresh_corr])
    index_greater_thresh = np.where(np.reshape(thresh_greater_corr, (-1, 1)) == p_corr)[
        1
    ]
    array_greater = X[:, index_greater_thresh]

    # Find less than thresh
    thresh_less_corr = np.array([corr for corr in p_corr if corr < thresh_corr])
    index_less_thresh = np.where(np.reshape(thresh_less_corr, (-1, 1)) == p_corr)[1]

    # Change type less than
    array_less = X[:, index_less_thresh]
    encoder = OneHotEncoder()
    array_less = encoder.fit_transform(array_less).toarray()

    output_x = np.concatenate((array_greater, array_less), axis=1)

    return output_x
