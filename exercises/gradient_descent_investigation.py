import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type
from itertools import product
from IMLearn.model_selection import cross_validate
from IMLearn.metrics import misclassification_error
from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    vals = []
    weights = []

    def callback(solver, weight, val, grad, t, eta, delta):
        vals.append(val)
        weights.append(weight)
    return callback,vals,weights
        


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    fig1 = go.Figure()
    for_fig = 0
    for loss,eta in product([L1,L2],etas):
        callback,v,w = get_gd_state_recorder_callback()
        w.append(init)
        learn_rate = FixedLR(eta)
        func = loss(init)
        gd = GradientDescent(learning_rate=learn_rate,callback=callback)
        gd.fit(func,None,None)
        fig = plot_descent_path(loss,np.array(w),title=f"eta equal to {eta}")
        fig.write_image(f"/Users/elkysandor/Desktop/hujiyr3/IML/plots_iml/plot_Q1.1.{for_fig}_ex6.png")
        fig1.add_trace(go.Scatter(x=list(range(len(v))), y=v, name= str(loss)[-4:-2] + f" convergence rate with eta = {eta}"))
        for_fig+=1
    fig1.update_xaxes(title="iteration")
    fig1.update_yaxes(title="loss value")
    fig1.write_image(f"/Users/elkysandor/Desktop/hujiyr3/IML/plots_iml/plot_Q1.20_ex6.png")





def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    for_fig = 0
    fig = go.Figure()
    for loss,gamma in product([L1],gammas):
        callback,v,w = get_gd_state_recorder_callback()
        w.append(init)
        learn_rate = ExponentialLR(eta,gamma)
        func = loss(init)
        v.append(func.compute_output())
        gd = GradientDescent(learning_rate=learn_rate,callback=callback)
        gd.fit(func,None,None)
        fig.add_trace(go.Scatter(x=list(range(len(v))), y=v, name=f"convergence rate with gamma = {gamma}"))
    fig.update_xaxes(title="iteration")
    fig.update_yaxes(title="loss value")
    fig.write_image(f"/Users/elkysandor/Desktop/hujiyr3/IML/plots_iml/plot_Q2.1_ex6.png")
    # Plot algorithm's convergence for the different values of gamma
    # raise NotImplementedError()

    # Plot descent path for gamma=0.95
    callback,v,w = get_gd_state_recorder_callback()
    w.append(init)
    learn_rate = ExponentialLR(eta,gammas[1])
    func = L1(init)
    gd = GradientDescent(learning_rate=learn_rate,callback=callback)
    gd.fit(func,None,None)
    fig = plot_descent_path(L1,np.array(w))
    fig.write_image(f"/Users/elkysandor/Desktop/hujiyr3/IML/plots_iml/plot_Q2.2_ex6.png")


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    # Plotting convergence rate of logistic regression over SA heart disease data
    logostic_clf = LogisticRegression()
    logostic_clf.fit(X_train.values, y_train.values)
    y_prob = logostic_clf.predict_proba(X_train.values)
    fpr, tpr, thresholds = roc_curve(y_train, y_prob)
    fig = go.Figure(
        data=[go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(color="black", dash='dash'), name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines',text=thresholds, name="", showlegend=False, marker_size=5,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    fig.write_html(f"/Users/elkysandor/Desktop/hujiyr3/IML/plots_iml/plot_Q3.1_ex6.html")
    # fig.write_image(f"/Users/elkysandor/Desktop/hujiyr3/IML/plots_iml/plot_Q3.1_ex6.png")
    alpha_hat = thresholds[(tpr - fpr).argmax()]
    logostic_clf2 = LogisticRegression(alpha=alpha_hat)
    logostic_clf2.fit(X_train.values, y_train.values)
    error = logostic_clf2.loss(X_test.values,y_test.values)
    print(f"the best alpha is {round(alpha_hat,4)} \n with error of {round(error,4)}")



    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lamdas = [0.001,0.002,0.005,0.01,0.02,0.05,0.1]
    range_lamdas = len(lamdas)
    holder = np.zeros((len(lamdas),2))
    for i,lamb in enumerate(product(["l1","l2"],lamdas)):
        logostic_clf_reg_l1 = LogisticRegression(penalty=lamb[0],lam=lamb[1])
        train_score,val_score = cross_validate(logostic_clf_reg_l1,X_train.values,y_train.values,misclassification_error)
        holder[i%range_lamdas,:] = train_score,val_score
        if i%range_lamdas == (range_lamdas-1):
            lambda_hat = lamdas[(holder[:,1]).argmin()]
            best_logostic_clf_reg = LogisticRegression(penalty=lamb[0],lam=lambda_hat)
            best_logostic_clf_reg.fit(X_train.values,y_train.values)
            errors = best_logostic_clf_reg.loss(X_test.values,y_test.values)
            print(f"the best lambda is {lambda_hat} with " + lamb[0] + " penalty "+ f" with error of {errors}")




if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
