from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        ndx = np.argsort(y)
        self.classes_, pos, class_count = np.unique(y[ndx],
                                                    return_index=True,
                                                    return_counts=True)

        class_sum = np.add.reduceat(X[ndx], pos, axis=0)
        self.pi_ = class_count[:,None]/X.shape[0]
        self.mu_ = class_sum / class_count[:,None]
        nomalize_mean = (X[ndx]-np.repeat(self.mu_, class_count, axis=0))**2
        nomalize_mean2 = (np.add.reduceat(nomalize_mean, pos, axis=0))
        self.vars_ = nomalize_mean2/class_count[:,None]
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        post_dist_per_k = np.zeros((X.shape[0],self.classes_.size))
        for i in range(self.classes_.size):
            resid = (X - self.mu_[i])**2
            sigma = self.vars_[i]
            def calc_post_dist(row):
                return -0.5*(np.log(2*np.pi*sigma)+(row/sigma))
            post_per_feature = np.apply_along_axis(calc_post_dist, 1, resid)
            post_dist_per_k[:,i] = (post_per_feature.sum(axis=1)+np.log(self.pi_[i]))
        return post_dist_per_k.argmax(axis=1)



    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        def calc_post_dist(sample_idx,class_idx):
            likelihood = 1/np.sqrt(2*np.pi)*1/(self.vars_[class_idx,:])*\
                         (np.exp(-0.5*((X[sample_idx,:]-self.mu_[class_idx,:])**2)/
                                 self.vars_[class_idx,:]))*self.pi_[class_idx]
            return likelihood
        classes_grid,feature_grid = np.meshgrid(np.arange(X.shape[0], np.arange(self.classes_.size), indexing="ij"))
        post_per_class = np.prod(calc_post_dist(classes_grid,feature_grid),axis=1)
        return post_per_class







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
            from ...metrics import misclassification_error
            return misclassification_error(y,self._predict(X))
