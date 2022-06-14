import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    daily_temp = pd.read_csv(filename,parse_dates={"date": [2]})
    daily_temp["DayOfYear"] = daily_temp.date.dt.x
    daily_temp = daily_temp.loc[daily_temp.Temp > -20]
    return daily_temp

def plot_israel(df):
    df.loc[:,"Year"] = df["Year"].astype("string")
    israel_data = df.loc[(df.Country == "Israel")]
    fig = px.scatter(israel_data,x = "DayOfYear",y="Temp",color="Year",title="Temp in israel as function of day of year"
                     , labels={'x': 'DayOfYear', 'y':'Temp'})
    fig.write_image("/Users/elkysandor/Desktop/hujiyr3/IML/plots_iml/plot_Q3.2.2a_ex2.png")

def std_by_month(df):
    israel_data = df.loc[(df.Country == "Israel")]
    tmp_std_by_month = israel_data.groupby("Month").agg(np.std).Temp
    fig = px.bar(tmp_std_by_month,x=tmp_std_by_month.index,y=tmp_std_by_month,
                 title="std temperature by month", labels={'x': 'month', 'y':'std'})
    fig.write_image("/Users/elkysandor/Desktop/hujiyr3/IML/plots_iml/plot_Q3.2.2b_ex2.png")

def group_by_contry_and_year(df):
    grouped_data = df.groupby(["Country","Month"]).agg([np.mean, np.std]).Temp
    fig = px.line(grouped_data,x=grouped_data.index.get_level_values(1),
                  y='mean',color=grouped_data.index.get_level_values(0),error_y="std",
                  title="behavior by country"
    , labels={'x': 'month', 'y':'mean temperature'})
    fig.write_image("/Users/elkysandor/Desktop/hujiyr3/IML/plots_iml/plot_Q3.2.3_ex2.png")

def quastion4(df):
    loss_per_model = []
    israel_data = df.loc[((df.Country == "Israel") & (df.Temp > -20))]
    train_x,train_y,test_x,test_y = split_train_test(israel_data,israel_data["Temp"])
    for k in range(1,11):
        poly_fit = PolynomialFitting(k)
        poly_fit.fit(train_x["DayOfYear"],train_y.values)
        loss_per_model.append(round(poly_fit.loss(test_x["DayOfYear"],test_y.values),2))
    loss_per_model = pd.Series(loss_per_model,index=np.arange(1,11,1))
    print(loss_per_model)
    fig = px.bar(loss_per_model,x=loss_per_model.index,y=loss_per_model,title="loss value by k"
                 , labels={'x': 'poly fitting degree', 'y':'MSE'})
    fig.write_image("/Users/elkysandor/Desktop/hujiyr3/IML/plots_iml/plot_Q3.2.4_ex2.png")

def last_quastion(df):
    loss_per_model = []
    israel_data = df.loc[(df.Country == "Israel")]
    poly_fit = PolynomialFitting(5)
    poly_fit.fit(israel_data["DayOfYear"],israel_data["Temp"])
    rest_data = df.loc[~(df.Country == "Israel")]
    for country in rest_data.Country.unique():
        loss = poly_fit.loss((df.loc[df.Country == country])["DayOfYear"],(df.loc[df.Country == country])["Temp"])
        loss_per_model.append(loss)
    for_plot = pd.Series(loss_per_model,index=rest_data.Country.unique())
    fig = px.bar(for_plot,x=for_plot.index,y=for_plot,title="loss value by model fitted on israel"
                 , labels={'x': 'country', 'y':'MSE'})
    fig.write_image("/Users/elkysandor/Desktop/hujiyr3/IML/plots_iml/plot_Q3.2.5_ex2.png")










if __name__ == '__main__':
    np.random.seed(0)
    X = load_data('/Users/elkysandor/Desktop/hujiyr3/IML/IML.HUJI/datasets/City_Temperature.csv')
    plot_israel(X)
    std_by_month(X)
    group_by_contry_and_year(X)
    quastion4(X)
    last_quastion(X)
