from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import datetime

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    house = pd.read_csv(filename).dropna().reset_index()
    y = house.price.map(lambda x : np.abs(x))
    zip_dummies = pd.get_dummies(house.zipcode,drop_first=True)
    zip_dummies.rename((lambda x : "zip_code "+ str(x)),axis=1)
    house.date = (house.date.str.slice(stop = 8))
    def to_unix(time_str):
        if time_str != "0":
            return datetime.datetime.strptime(time_str, "%Y%m%d").timestamp()
        return 0
    house["date"] = house.date.apply(to_unix)
    house["age"] = 2016-house.yr_built
    q = pd.qcut(house["yr_renovated"].loc[house.yr_renovated!=0],4,labels=[1,2,3,4])
    last_renovation_lab = pd.concat([(house.yr_renovated.loc[house.yr_renovated==0]),q]).sort_index()
    renovation_dummies = pd.get_dummies(last_renovation_lab,drop_first=True)
    house = house.join(renovation_dummies)
    house = house.join(zip_dummies)
    house.drop(index=20667,inplace=True)
    y.drop(index=20667,inplace=True)
    house.drop(columns=["index","id","yr_built","yr_renovated","zipcode","long","price"],inplace=True)
    house.reset_index(inplace=True,drop=True)
    y = y.reset_index(drop=True)
    return house ,y



def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    X.columns = X.columns.astype(str)
    X_std = X.std()
    y_std = y.std()
    cov_with_response = np.cov(X.T,y.T)[:,-1]
    corr_vec = cov_with_response/((X_std.append(pd.Series({"y":y_std})))*y_std)
    for i,feature in enumerate(X.columns.astype(str)):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X[feature], y=y.values, mode="markers"))
        fig.update_layout({"title": f"{feature} <br><sup>pearson cor is {round(corr_vec[i],3)}<sup>",
                           "yaxis_title": "price"})
        fig.write_image(f"/Users/elkysandor/Desktop/hujiyr3/IML/plots_iml/ex2_many_plots/plot_Q2_ex2_{i}.png")

def qa_4(train_df,train_vec,test_df,test_vec):
    lin_reg = LinearRegression()
    precentage = np.arange(.10,1,0.01)
    mean_loss = pd.DataFrame(columns=['mean',"std"])
    for p in precentage:
        loss = np.zeros(10)
        for rep in range(10):
            sample = train_df.sample(frac=p)
            lin_reg.fit(sample,train_vec[sample.index])
            loss[rep] = lin_reg.loss(test_df,test_vec)
        mean_loss = mean_loss.append({"mean":loss.mean(),"std":loss.std()},ignore_index=True)
    fig = go.Figure([go.Scatter(x=precentage, y=mean_loss["mean"]-2*mean_loss["std"], fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False),
               go.Scatter(x=precentage, y=mean_loss["mean"]+2*mean_loss["std"], fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False),
               go.Scatter(x=precentage, y=mean_loss["mean"], mode="markers+lines", marker=dict(color="black",size=1), showlegend=False)],
              layout=go.Layout(title=f"mean and std as of loss as function of p",
                               height=300))
    fig.write_image(f"/Users/elkysandor/Desktop/hujiyr3/IML/plots_iml/plot_Q3.1.4_ex2.png")



if __name__ == '__main__':
    np.random.seed(0)
    X,y = load_data("/Users/elkysandor/Desktop/hujiyr3/IML/IML.HUJI/datasets/house_prices.csv")
    feature_evaluation(X,y)
    train_x,train_y,test_x,test_y = split_train_test(X,y)
    for i in (train_x,train_y,test_x,test_y):
        print(i.shape)
    qa_4(train_x,train_y,test_x,test_y)

