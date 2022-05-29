from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    result_lst = np.zeros((cv,2))
    samples_idx = np.array_split(np.arange(y.size),cv)
    for exclude,idx in enumerate(samples_idx):
        if exclude == cv-1:
            train_folds_idx = np.concatenate(samples_idx[:exclude]).flatten()
        elif exclude == 0:
            train_folds_idx = np.concatenate(samples_idx[exclude+1:]).flatten()
        else:
            pre_indexes = np.concatenate(samples_idx[:exclude]).flatten()
            post_indexs = np.concatenate(samples_idx[exclude+1:]).flatten()
            train_folds_idx = np.concatenate([pre_indexes,post_indexs],casting="unsafe",dtype="int")
        train_folds, y_folds = X[train_folds_idx,:], y[train_folds_idx]
        test_fold, y_fold = X[idx,:], y[idx]
        estimator.fit(train_folds,y_folds)
        y_train_pred, y_test_pred = estimator.predict(train_folds), estimator.predict(test_fold)
        result_lst[exclude,:] = np.array([scoring(y_train_pred,y_folds),scoring(y_test_pred,y_fold)])
    return result_lst.mean(axis=0)

