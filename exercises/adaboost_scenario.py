import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def decision_surface(predict,T,xrange, yrange, density=120, dotted=False, colorscale=custom, showscale=True):
    xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange, density)
    xx, yy = np.meshgrid(xrange, yrange)
    pred = predict(np.c_[xx.ravel(), yy.ravel()],T)

    if dotted:
        return go.Scatter(x=xx.ravel(), y=yy.ravel(), opacity=1, mode="markers", marker=dict(color=pred, size=1, colorscale=colorscale, reversescale=False), hoverinfo="skip", showlegend=False)
    return go.Contour(x=xrange, y=yrange, z=pred.reshape(xx.shape), colorscale=colorscale, reversescale=False, opacity=.7, connectgaps=True, hoverinfo="skip", showlegend=False, showscale=showscale)

def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump,n_learners).fit(train_X,train_y)
    train_err = np.array([adaboost.partial_loss(train_X,train_y,t) for t in range(1,n_learners+1)])
    test_err = np.array([adaboost.partial_loss(test_X,test_y,t) for t in range(1,n_learners+1)])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(250),y=train_err,name="train"))
    fig.add_trace(go.Scatter(x=np.arange(250),y=test_err,name="test"))
    fig.update_layout({"title":"errors as function of iteration",
                       "xaxis_title":"iteration","yaxis_title":"normalized error"})
    fig.write_image(f"/Users/elkysandor/Desktop/hujiyr3/IML/plots_iml/plot_Q1_ex4_{noise}.png")
    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    T_idx = [4,49,99,249]
    fig2 = make_subplots(rows=2, cols=2,
            subplot_titles=[f"{t} learners with error of {test_err[t-1]}" for t in T],
            horizontal_spacing = 0.01, vertical_spacing=.03)
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T
    for i in range(4):
        fig2.add_traces([decision_surface(adaboost.partial_predict,T[i], lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:,0], y=test_X[:,1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i//2)+1, cols=(i%2)+1)
    fig2.write_image(f"/Users/elkysandor/Desktop/hujiyr3/IML/plots_iml/plot_Q2_ex4_{noise}.png")

    # Question 3: Decision surface of best performing ensemble
    best_learner = test_err.argmin()
    acc = (adaboost.partial_predict(test_X,best_learner)==test_y).mean()
    fig = go.Figure()
    fig.add_traces([decision_surface(adaboost.partial_predict,best_learner, lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:,0], y=test_X[:,1], mode="markers", showlegend=False,
                               marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1)))])
    fig.update_layout({"title":f"ensemble size is {best_learner+1} with {round(acc,3)} accuracy"})
    fig.write_image(f"/Users/elkysandor/Desktop/hujiyr3/IML/plots_iml/plot_Q3_ex4_{noise}.png")


    # Question 4: Decision surface with weighted samples
    fig = go.Figure()
    s = (adaboost.D_/adaboost.D_.max()) * 5
    fig.add_traces([decision_surface(adaboost.partial_predict,250, lims[0], lims[1], showscale=False),
                    go.Scatter(x=train_X[:,0], y=train_X[:,1], mode="markers", showlegend=False,
                               marker=dict(color=train_y, colorscale=[custom[0], custom[-1]],size=s,
                                           line=dict(color="black", width=1)))])
    fig.update_layout({"title" : "adaboost with distribution"})
    fig.write_image(f"/Users/elkysandor/Desktop/hujiyr3/IML/plots_iml/plot_Q4_ex4_{noise}.png")


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
