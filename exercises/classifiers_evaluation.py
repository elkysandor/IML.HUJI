from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    data_dict = {}
    counter = 1
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X,y_all = load_dataset(f"/Users/elkysandor/Desktop/hujiyr3/IML/IML.HUJI/datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        def default_callback(fit: Perceptron, x: np.ndarray, y: int):
            loss = fit.loss(X,y_all)
            losses.append(loss)
        perceptron = Perceptron(callback=default_callback)
        perceptron.fit(X,y_all)
        # Plot figure of loss as function of fitting iteration
        fig = go.Figure(go.Scatter(x=[i+1 for i, j in enumerate(losses)], y=losses,mode="lines+markers"))
        fig.update_layout({"title":f"loss as function of iteration on {n} data",
                           "xaxis_title":"iteration","yaxis_title":"loss"})
        fig.write_image(f"/Users/elkysandor/Desktop/hujiyr3/IML/plots_iml/plot_Q3.1.{counter}_ex3.png")
        counter+=1

def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    symbols = np.array(["circle", "x","star"])
    models = ["LDA", "GNB"]
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X,y = load_dataset(f"/Users/elkysandor/Desktop/hujiyr3/IML/IML.HUJI/datasets/{f}")
        from sklearn.naive_bayes import GaussianNB
        # Fit models and predict over training set
        lda = LDA().fit(X,y)
        nb = GaussianNaiveBayes().fit(X,y)
        gnb = GaussianNB()
        gnb.fit(X,y)
        y_pred=[lda.predict(X),nb.predict(X)]

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions

        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy

        fig = make_subplots(rows=1, cols=2,subplot_titles=[f"{models[i]} classifier with accuracy {round(accuracy(y,m),4)}" for i,m in enumerate(y_pred)])
        fig.add_trace(
            go.Scatter(x=X[:,0], y=X[:,1], mode = 'markers',showlegend=False,
                       marker = dict(color=y_pred[0], symbol=symbols[y],
                                     colorscale=[[0.0, 'rgb(165,0,38)'], [1.0, 'rgb(49,54,149)']],
                                     reversescale=True, size = 6,)),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(x=X[:,0], y=X[:,1], mode = 'markers',showlegend=False ,
                       marker = dict(color=y_pred[1],symbol=symbols[y], colorscale=[[0.0, 'rgb(165,0,38)'], [1.0, 'rgb(49,54,149)']], reversescale=True, size = 6)),
            row=1, col=2)
        fig.update_layout(height=600, width=800)
        fig.write_image(f"/Users/elkysandor/Desktop/hujiyr3/IML/plots_iml/plot_Q3.2.3_ex3{f}.png")

        # Add traces for data-points setting symbols and colors
        # raise NotImplementedError()

        # Add `X` dots specifying fitted Gaussians' means
        # raise NotImplementedError()

        # Add ellipses depicting the covariances of the fitted Gaussians
        # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
