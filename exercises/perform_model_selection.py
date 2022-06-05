from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    f = lambda x: (x+3)*(x+2)*(x+1)*(x-1)*(x-2)
    random_x = np.random.uniform(-1.2,2,n_samples)
    fuss = np.random.normal(0,noise,n_samples)
    dat = f(random_x)
    noise_dat = f(random_x)+fuss
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(random_x), pd.Series(noise_dat),2/3)
    fig = go.Figure()
    fig.add_traces([go.Scatter(x=random_x,y=dat,mode="markers",marker={"color":"green"},name="oracle"),
                    go.Scatter(x=train_X[0],y=train_y,mode="markers",marker={"color":"blue"},name="train"),
                    go.Scatter(x=test_X[0],y=test_y,mode="markers",marker={"color":"red"},name="test")])
    fig.write_image(f"/Users/elkysandor/Desktop/hujiyr3/IML/plots_iml/plot_Q2.1.1.{noise}_{n_samples}_ex5.png")



    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    cors_valid_scr = np.zeros((11,2))
    for k in range(11):
        polyfit = PolynomialFitting(k)
        train_score,validation_score = cross_validate(polyfit,train_X.values,train_y.values,mean_square_error)
        cors_valid_scr[k,:] = train_score,validation_score
    fig = go.Figure()
    fig.add_traces([go.Bar(x=np.arange(12),
        y=cors_valid_scr[:,0],
        text=cors_valid_scr[:,0].round(2),
        textposition='auto',
        name='train mse',
        marker_color="mediumseagreen"),
        go.Bar(x=np.arange(12),
        y=cors_valid_scr[:,1],
        text=cors_valid_scr[:,1].round(2),
        textposition='auto',
        name='validation mse',
        marker_color='lightsalmon')])
    fig.update_layout(barmode='group')
    fig.write_image(f"/Users/elkysandor/Desktop/hujiyr3/IML/plots_iml/plot_Q2.1.2.{noise}_{n_samples}_ex5.png")


    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_hat = cors_valid_scr[:,1].argmin()
    final_ploy = PolynomialFitting(k_hat)
    final_ploy.fit(train_X.values,train_y.values)
    y_pred = final_ploy.predict(test_X.values)
    print(f" k is {k_hat} and error is {round(mean_square_error(test_y.values,y_pred),2)}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X,y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y, test_X, test_y = X[:n_samples,:], y[:n_samples], X[n_samples:,:], y[n_samples:]



    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lam_options = np.linspace(0.001,2,n_evaluations)
    results = np.zeros((n_evaluations,4))
    for num,lamb in enumerate(lam_options):
        ridge = RidgeRegression(lam=lamb)
        lasso = Lasso(alpha=lamb)
        rg_train_score,rg_validation_score = cross_validate(ridge,train_X,train_y,mean_square_error)
        ls_train_score,ls_validation_score = cross_validate(lasso,train_X,train_y,mean_square_error)
        rg_y_test_pred = (ridge.fit(train_X,train_y)).predict(test_X)
        ls_y_test_pred = (lasso.fit(train_X,train_y)).predict(test_X)
        results[num,:] = rg_train_score, rg_validation_score, ls_train_score, ls_validation_score
    models = ["ridge regression","lasso regression"]
    fig = make_subplots(rows=1, cols=2,subplot_titles=models,)
    fig.add_traces([go.Scatter(x=lam_options,y=results[:,0],name="train error"),
                    go.Scatter(x=lam_options,y=results[:,1],name="validation error")],
                   rows=1, cols=1)
    fig.add_traces([go.Scatter(x=lam_options,y=results[:,2],name="train error"),
                    go.Scatter(x=lam_options,y=results[:,3],name="validation error")],
                   rows=1, cols=2)
    fig.write_image(f"/Users/elkysandor/Desktop/hujiyr3/IML/plots_iml/plot_Q2.2.7._ex5.png")


    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_rg_lambda = lam_options[results[:,1].argmin()]
    best_ls_lambda = lam_options[results[:,3].argmin()]
    print(f" rg {best_rg_lambda} ls {best_ls_lambda}")
    ridge = RidgeRegression(lam=best_rg_lambda)
    lasso = Lasso(alpha=best_ls_lambda)
    ols = LinearRegression()
    ridge.fit(train_X,train_y)
    lasso.fit(train_X,train_y)
    ols.fit(train_X,train_y)
    rg_error = round(mean_square_error(test_y,ridge.predict(test_X)),3)
    ls_error = round(mean_square_error(test_y,lasso.predict(test_X)),3)
    ols_error = round(mean_square_error(test_y,ols.predict(test_X)),3)
    print(f" the ridge error is {rg_error} and lasso error is {ls_error} and ols error is {ols_error}")
if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500,noise=10)
    select_regularization_parameter()
