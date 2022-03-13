from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=True

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = X.mean()
        self.var_ = X.var(ddof=int(not self.biased_))
        # raise NotImplementedError()

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        pdf = lambda x: (1/np.sqrt(2*np.pi*self.var_)) * np.exp((np.power(x-self.mu_, 2))/(-2*self.var_))
        return pdf(X)
        # raise NotImplementedError()

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        raise NotImplementedError()


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.ft`
            function.

        cov_: float
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.ft`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = X.mean(axis=0)
        self.cov_ = np.cov(np.random.multivariate_normal([1,2],[[1,0],[0,1]],6), rowvar=False)
        # raise NotImplementedError()

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        raise NotImplementedError()

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        cov : float
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        raise NotImplementedError()

def plot_question_2(diff: np.ndarray):
    df_for_plot = pd.DataFrame({"size": np.arange(10,1010,10), "differnce": diff})
    fig = px.line(df_for_plot, x="size", y="differnce",range_x=[0,1030],
                  title="L1 norm from true expected as function on sample size")
    fig.update_xaxes(nticks=20)
    fig.update_traces(line_color="#00ff00")
    fig.update_xaxes(title_text="size of sample")
    fig.update_yaxes(title_text="L1 distance from true expected")
    fig.write_image("/Users/elkysandor/Desktop/plots_iml/plot_Q2.png")

def plot_question_3(pdf_vec):
    df_for_plot = pd.DataFrame({"x": np.arange(1,1001,1), "pdf": pdf_vec})
    fig = px.scatter(df_for_plot, x="x", y="pdf",range_x=[0,1005],
                  title="empirical PDF")
    fig.update_xaxes(nticks=20)
    fig.update_traces(line_color="#00ff00")
    fig.update_xaxes(title_text="rank of sample")
    fig.update_yaxes(title_text="pdf of sample")
    fig.write_image("/Users/elkysandor/Desktop/plots_iml/plot_Q3.png")


if __name__ == '__main__':
    # Q1
    univ_gaussian = UnivariateGaussian()
    np.random.seed(1)
    rndm_normal = np.random.normal(10,1,1000)
    univ_gaussian.fit(rndm_normal)
    print(f" the expectation and variance are {(univ_gaussian.mu_,univ_gaussian.var_)}")

    # #Q2
    # np.random.sample()
    # expected_per_n = np.zeros(np.arange(10,1010,10).shape)
    # for i,n in enumerate(np.arange(10,1010,10)):
    #     sample_of_size_n = np.random.choice(rndm_normal,n)
    #     univ_gaussian.fit(sample_of_size_n)
    #     expected_per_n[i] = univ_gaussian.mu_
    # abs_dist = np.abs(expected_per_n - 10)
    # plot_question_2(abs_dist)

    #Q3
    univ_gaussian.fit(rndm_normal)
    plot_question_3(univ_gaussian.pdf(rndm_normal))




