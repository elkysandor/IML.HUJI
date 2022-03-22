from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"

def plot_question_2(diff: np.ndarray):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(10,1010,10), y=diff,mode="lines+markers"))
    fig.update_layout({"title":"L1 norm from true expected as function of sample size",
                       "xaxis_title":"size of sample","yaxis_title":"L1 distance from true expected"})
    fig.update_xaxes(nticks=20)
    fig.write_image("/Users/elkysandor/Desktop/plots_iml/plot_Q2.png")


def plot_question_3(pdf_vec,samples):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=samples, y=pdf_vec, mode="markers"))
    fig.update_layout({"title": "empirical PDF", "xaxis_title": "samples value",
                       "yaxis_title": "pdf of sample"})
    fig.update_xaxes(nticks=20)
    fig.write_image("/Users/elkysandor/Desktop/plots_iml/plot_Q3.png")


def plot_heatmap(mat):
    fig = go.Figure()
    f = np.linspace(-10,10,200)
    fig.add_trace(go.Heatmap(z=mat,x=f,y=f,colorscale="YlOrRd"))
    fig.update_layout({"title": "heat map", "xaxis_title": "f3 values",
                       "yaxis_title": "f1 values"})
    fig.write_image("/Users/elkysandor/Desktop/plots_iml/heatmap.png")

def test_univariate_gaussian():
    univ_gaussian = UnivariateGaussian()
    rndm_normal = np.random.normal(10,1,1000)
    univ_gaussian.fit(rndm_normal)
    print(f" the expectation and variance are {(univ_gaussian.mu_,univ_gaussian.var_)}")
    expected_per_n = np.zeros(np.arange(10,1010,10).size)
    for i,n in enumerate(np.arange(10,1010,10)):
        sample_of_size_n = rndm_normal[:n]
        univ_gaussian.fit(sample_of_size_n)
        expected_per_n[i] = univ_gaussian.mu_
    abs_dist = np.abs(expected_per_n - 10)
    plot_question_2(abs_dist)
    univ_gaussian.fit(rndm_normal)
    plot_question_3(univ_gaussian.pdf(rndm_normal),rndm_normal)



def test_multivariate_gaussian():
    mu = np.array([0,0,4,0])
    cov_mat = np.array([[1,0.2,0,0.5],[0.2,2,0,0],[0,0,1,0],[0.5,0,0,1]])
    rndm_multi_normal = np.random.multivariate_normal(mu, cov_mat,1000)
    multinormal = MultivariateGaussian()
    multinormal.fit(rndm_multi_normal)
    print(f"the expected vector is {multinormal.mu_} \nthe covariance matrix is \n{multinormal.cov_}")
    f = np.linspace(-10,10,200)
    Likelihood_mat = np.zeros((200,200))
    for i,f1 in enumerate(f):
        for j,f3 in enumerate(f):
            Likelihood_mat[i,j] = MultivariateGaussian.log_likelihood(np.array([f1,0,f3,0]),cov_mat,rndm_multi_normal)
    plot_heatmap(Likelihood_mat)

    index_mat = np.unravel_index(Likelihood_mat.argmax(),Likelihood_mat.shape)
    print(f" the best values of f1 and f3 are {round(f[index_mat[0]],3),round(f[index_mat[1]],3)}")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
