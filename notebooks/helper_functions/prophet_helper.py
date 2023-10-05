import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error
from typing import Tuple


def isolate_stasiun(
    df:pd.DataFrame,
    used_stasiun:str,
    time_col:str='read_time',
    y_col:str='pm25'
    ) -> pd.DataFrame:
    """select a certain stasiun from klhk aqi data according to the data required
    by prophet

    Args:
        df (pd.DataFrame): dataframe with datetime, aqi, and stasiun data
        used_stasiun (str): name of the stasiun to be used
        time_col (str, optional): the datetime column name. Defaults to 'read_time'.
        y_col (str, optional): the aqi column name. Defaults to 'pm25'.

    Returns:
        stasiun_data (pd.DataFrame): formated and filtered df
    """
    stasiun_data = df[df['stasiun'] == used_stasiun]
    stasiun_data = (
        stasiun_data[[time_col, y_col]]
        .rename(columns={
            time_col:'ds',
            y_col:'y'
        })
        )
    return stasiun_data

def plot_test(
    stasiun_name:str,
    train_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    test_df: pd.DataFrame
    ):
    """draw a plot of train, test, and forecasted data along with it's test rmse

    Args:
        stasiun_name (str): name of the stasiun to be plotted
        train_df (pd.DataFrame): training dataframe
        forecast_df (pd.DataFrame): forecasted dataframe
        test_df (pd.DataFrame): testing dataframe

    Returns:
        _type_: _description_
    """
    
    # Create a Matplotlib figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot training data, true prediction value, and predicted value
    ax.plot(train_df['ds'], train_df['y'], label='Training Data')
    ax.plot(test_df['ds'], test_df['y'], label='True Prediction Value')
    ax.plot(forecast_df['ds'], forecast_df['yhat'], label='Predicted Value')

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_df['y'], forecast_df['yhat']))

    # Add title and labels
    ax.set_title(
        f'{stasiun_name} Training Data vs. True Prediction vs. Predicted Value'
        )
    ax.set_xlabel('Datetime')
    ax.set_ylabel('PM 2.5 Index')

    # Add RMSE as a subtitle
    ax.annotate(f'RMSE: {rmse:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=12, color='red')

    # Add legend
    ax.legend()

    return fig, rmse

def plotly_plot_test(
    train_df:pd.DataFrame,
    forecast_df:pd.DataFrame,
    test_df:pd.DataFrame
):
    """_summary_

    Args:
        train_df (pd.DataFrame): _description_
        forecast_df (pd.DataFrame): _description_
        test_df (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    # Create traces for training data, true prediction value, and predicted value
    trace_train = go.Scatter(
        x=train_df['ds'], y=train_df['y'], mode='lines', name='Training Data'
        )
    trace_true_pred = go.Scatter(
        x=test_df['ds'], y=test_df['y'], mode='lines', name='True Prediction Value'
        )
    trace_pred = go.Scatter(
        x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Predicted Value'
        )

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_df['y'], forecast_df['yhat']))

    # Create layout for the plot with RMSE as subtitle
    layout = go.Layout(title='Training Data vs. True Prediction vs. Predicted Value',
                    xaxis=dict(title='Datetime'),
                    yaxis=dict(title='Value'),
                    annotations=[dict(xref='paper', yref='paper', x=0, y=1.1,
                                        text=f'RMSE: {rmse:.4f}', showarrow=False,
                                        font=dict(color='red'))])

    # Create a figure and add the traces
    fig = go.Figure(data=[trace_train, trace_true_pred, trace_pred], layout=layout)

    return fig, rmse

# Python
def warm_start_params(m):
    """
    Retrieve parameters from a trained model in the format used to initialize a new Stan model.
    Note that the new Stan model must have these same settings:
        n_changepoints, seasonality features, mcmc sampling
    for the retrieved parameters to be valid for the new model.

    Parameters
    ----------
    m: A trained model of the Prophet class.

    Returns
    -------
    A Dictionary containing retrieved parameters of m.
    """
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        if m.mcmc_samples == 0:
            res[pname] = m.params[pname][0][0]
        else:
            res[pname] = np.mean(m.params[pname])
    for pname in ['delta', 'beta']:
        if m.mcmc_samples == 0:
            res[pname] = m.params[pname][0]
        else:
            res[pname] = np.mean(m.params[pname], axis=0)
    return res
