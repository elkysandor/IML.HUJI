from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        counter1 = np.inf
        for sign,col in product([-1,1],range(X.shape[1])):
            trr,trr_err = self._find_threshold(X[:,col],y,sign)
            if trr_err < counter1:
                self.threshold_, self.j_, self.sign_ = trr, col, sign
                counter1 = trr_err




    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        y_hat = self.sign_ * np.sign((X[:,self.j_] >= self.threshold_)-0.5)
        return y_hat.astype(int)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # find_t = np.tile(values,(values.size,1))
        # new_labels = values[:,None] <= find_t
        # if sign == -1:
        #     new_labels = ~new_labels
        # new_labels2 = np.sign(new_labels.astype(int)-0.5).astype(int)
        # Misclassificaiton_vec = ((new_labels2 != np.sign(labels)) * np.abs(labels)).mean(axis=1)
        # return values[Misclassificaiton_vec.argmin()],  Misclassificaiton_vec.min()
        sample_size = values.size
        ord_idx = np.argsort(values)
        cumsum = np.cumsum(labels[ord_idx] * sign)
        shifted_cs = np.zeros(sample_size)
        shifted_cs[1:] = cumsum[:-1]
        sim = -(np.full(sample_size,cumsum[-1]) - 2*shifted_cs)
        trr_idx = np.argmin(sim)
        return values[ord_idx][trr_idx], sim[trr_idx]


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return (self._predict(X) != np.sign(y)).mean()
