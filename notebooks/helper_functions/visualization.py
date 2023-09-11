"""Module that contain functions to help with visualization
"""
import random
from typing import Callable, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flexitext import flexitext


def overall_plot(
    df_to_plot: pd.DataFrame,
    x: str,
    y: List,
    ncols:int
    ) -> Tuple[plt.Figure, plt.Axes]:
    """Create an overall timeseries line graph

    Args:
        df_to_plot (pd.DataFrame): the dataframe to be plotted
        x (str): column name of the x axis (timeseries)
        y (List): list of column names to be plotted
        ncols (int): number of column of the plot/figure

    Returns:
        Tuple[plt.Figure, plt.Axes]
    """
    nrow = len(y)

    fig, ax = plt.subplots(
        nrow,
        ncols,
        figsize = (15,4*nrow)
        )

    if (ncols>1) or (nrow>1):
        for y, ax in zip(y, ax.flatten()):
            ax.plot(
                df_to_plot[x],
                df_to_plot[y[0]],
                linestyle='-',
                )
            ax.set_title(f'{y.upper()} Levels Over Time', fontweight = "bold", fontsize = 20, fontfamily = "monospace")
            ax.set_xlabel('Time')
            ax.set_ylabel(f'{y}')
    else:
        ax.plot(
            df_to_plot[x],
            df_to_plot[y[0]],
            linestyle='-',
            )
        ax.set_title(f'{y[0].upper()} Levels Over Time', fontweight = "bold", fontsize = 20, fontfamily = "monospace")
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{y[0]}')

    plt.tight_layout()
    
    return fig, ax

def timely_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    time_granularity: str='day',
    nrow: int=4,
    ncol: int=2
):
    """plot timely trend of the specified variable.

    Args:
        df (pd.DataFrame): the dataframe to be plotted
        x (str): timeseries column
        y (str): variable to be plotted
        time_granularity (str): the time granularity, Defaults to 'day',
        nrow (int, optional): number of row. Defaults to 4.
        ncol (int, optional): number of column. Defaults to 2.
    """
    # initiate fig and ax
    fig, ax = plt.subplots(
        nrow,
        ncol,
        figsize=(14,2.5*nrow),
        sharey=True
        )
    if time_granularity=='day':
        rand_date = sorted(random.sample(
            df[x].dt.date.unique().tolist(), 
            nrow*ncol
        ))
        date_fmt = mdates.DateFormatter('%H:%M:%S')  # Format for the x-axis labels
    elif time_granularity=='week':
        rand_date = sorted(random.sample(
            df[x].dt.strftime('%Y-%U').unique().tolist(),
            nrow*ncol
        ))
        date_fmt = mdates.DateFormatter("%A")  # Format for the x-axis labels
    else:
        rand_date = sorted(
            df[x].dt.strftime('%Y-%m').unique().tolist()
        )
        date_fmt = mdates.DateFormatter('%m-%d')  # Format for the x-axis labels

    for date, ax in zip(rand_date, ax.flatten()):
        if time_granularity=='day':
            data = df[df[x].dt.date == date]
        elif time_granularity=='week':
            data = df[df[x].dt.strftime('%Y-%U') == date]
        else:
            data = df[df[x].dt.strftime('%Y-%m') == date]
        # Plotting with time on the x-axis
        ax.plot(
            data[x],
            data[y],
        )
        ax.xaxis.set_major_formatter(date_fmt)  # Set the x-axis format to show time only
        ax.set_title(f'{date}', fontweight="bold", fontsize=14, fontfamily="monospace")
        
    title = (
        f"<name:monospace, size:20> The trend of hourly {y} on {nrow*ncol} random "
        f"<color: #d43535>{time_granularity}</></>"
    )
    flexitext(0.5, 1.03, title, va="top", ha="center", xycoords="figure fraction")
    
    plt.tight_layout()

def reg_line(
    x: np.array,
    y: np.array
    ) -> Tuple[float, float, np.array]:
    """Fit linear regression via least squares with numpy.polyfit
    deg=1 means linear fit (i.e. polynomial of degree 1)

    Args:
        x (np.array): array of independent variables
        y (np.array): array of dependent variables

    Returns:
        Tuple[float, float, np.array]: slope, intercept, and the fitted
        line array
    """
    slope, intercept = np.polyfit(x, y, deg=1)
    # Create sequence for plotting
    xseq = np.linspace(min(x), max(x))
    return slope, intercept, xseq

def plot_scatter_with_reg(
    data:pd.DataFrame,
    xs:List,
    ys:str,
    nrow:int,
    ncol:int,
    reg_func:Callable,
    norm_func:Callable=None
    ):
    """Create a small multiple scatter plot with regression line
    The number of the graph returned will be nrow*ncol

    Args:
        data (pd.DataFrame): the dataframe to be plotted
        xs (List): list of item from the dataframe columns which will be the x axis of the plots
        y (str): one string item from the dataframe columns which will be the y axis of the plots
        nrow (int): number of row of the visualization
        ncol (int): number of columns of the visualization
        reg_func (Callable): regression function that will be used to be fit into the data to make the regression line
        norm_func (Callable): normalization function that will be used to be fit into the data
    """
    fig, axs = plt.subplots(nrow,ncol,figsize = (15,3*nrow), sharey=True)

    plt.tight_layout()

    for col, ax in zip(xs, axs.flatten()):
        if norm_func:
            x = norm_func(data[col])
            y = norm_func(data[ys])
        else:
            x = data[col]
            y = data[ys]

        # Add scatterplot
        ax.scatter(x, y, s=10, alpha=0.7, edgecolors="k")
        ax.set_xlabel(f"{col}")
        ax.set_ylabel(ys)

        try:
            slope, intercept, xseq = reg_func(x,y)

            # Plot regression line
            ax.plot(
                xseq,
                slope * xseq + intercept,
                color="red",
                lw=2.5,
                label='y={:.2f}x+{:.2f}'.format(slope,intercept)
                )
            ax.legend(fontsize=9)
        except:
            pass

    plt.show()

def plot_lagged_scatter(
    data:pd.DataFrame,
    reg_func:Callable=reg_line,
    shift:int=-1,
    parameter:str="pm2.5",
    ):
    temp_df = data[[parameter]]
    lagged_param_name = parameter+"_lag"
    temp_df = temp_df.assign(**{lagged_param_name: temp_df[parameter].shift(shift)})
    x=temp_df[parameter]
    y=temp_df[lagged_param_name]
    
    try:
        slope, intercept, xseq = reg_func(x.iloc[:shift],y[:shift])
    except:
        x=x.ffill()
        y=y.ffill()
        slope, intercept, xseq = reg_func(x.iloc[:shift],y[:shift])

    fig, ax = plt.subplots()
    # plot points
    ax.scatter(
        x=x,
        y=y,
        edgecolors="k",
        s=10
    )

    # Plot regression line
    ax.plot(
        xseq, 
        slope * xseq + intercept, 
        color="red", 
        lw=2.5,
        label='y={:.2f}x+{:.2f}'.format(slope,intercept)
        )
    ax.legend(fontsize=9)
    
    plt.show()

def plot_correlation_heatmap(dataframe:pd.DataFrame):
    """generate a correlation heatmap matrix

    Args:
        dataframe (pd.DataFrame): the dataframe to be plotted
    """
    # Calculate pairwise-correlation
    matrix = dataframe.corr()

    # Create a mask
    mask = np.triu(np.ones_like(matrix, dtype=bool))

    # Create a custom diverging palette
    cmap = sns.diverging_palette(
        250, 15, s=75, l=40, n=9,
        center="light",
        as_cmap=True
    )

    plt.figure(figsize=(16, 12))

    sns.heatmap(matrix, mask=mask, center=0, annot=True,
                fmt='.2f', square=True, cmap=cmap)

    plt.show()

def plot_prediction(
    length:int,
    train_data:pd.DataFrame,
    test_data:pd.DataFrame,
    predictions:pd.DataFrame,
    save:bool=False,
    img_save_dir:str=None
):
    """Plot train, test, and prediction data

    Args:
        length (int): the length of data to be drawn
        train_data (pd.DataFrame): train data df
        test_data (pd.DataFrame): test data df
        predictions (pd.DataFrame): prediction df
        save (bool, optional): save the plot or not. Defaults to False.
    """
    temp_train = (
        train_data[['pm2.5', 'read_time']].reset_index()
        .tail(length).reset_index()
        .drop(columns=["index", "level_0"])
        .rename(columns={'pm2.5':'pm2.5_train'})
    )
    temp_test = (
        test_data[['pm2.5', 'read_time']].reset_index()
        .head(length).reset_index()
        .drop(columns=["index", "level_0"])
        .rename(columns={'pm2.5':'pm2.5_test'})
    )

    temp_pred = predictions.head(length).rename(columns={'pm2.5':'pm2.5_prediction'})

    temp_pred_test = pd.merge(temp_pred, temp_test, on='read_time', how='inner')

    # Calculate MSE and RMSE
    temp_pred_test['squared_error'] = (temp_pred_test['pm2.5_test'] - temp_pred_test['pm2.5_prediction']) ** 2
    # mse = temp_pred_test['squared_error'].mean()
    rmse = (temp_pred_test['squared_error'].mean())**.5

    _, ax = plt.subplots(figsize = (20,5))

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.xlabel("")

    temp_train.plot(
        x="read_time",
        y="pm2.5_train",
        ax=ax,
        label="Train",
        color="blue",
        lw=2
    )

    temp_pred_test.plot(
        x="read_time",
        y="pm2.5_prediction",
        ax=ax,
        label="Prediction",
        color="red",
        marker="o",
        lw=3
    )

    # Define the confidence interval
    ci = 0.95 * (temp_pred_test['squared_error'].std() / np.sqrt(len(temp_pred_test)))

    ax.fill_between(
        temp_pred_test.read_time.values, 
        (temp_pred_test["pm2.5_prediction"]-ci).to_numpy(), 
        (temp_pred_test["pm2.5_prediction"]+ci).to_numpy(), 
        color='yellow', alpha=0.1,
        label=r"95% confidence interval"
    )

    temp_pred_test.plot(
        x="read_time",
        y="pm2.5_test",
        linestyle='--',
        ax=ax,
        label="True Value",
        color="black",
        marker="o",
        lw=2
    )

    title = "<name:monospace, size:18><weight:bold>Forecasting Result</></>"
    flexitext(0, 1.20, title, va="top", ax=ax)

    subtitle = (
        f"<name:monospace, size:12, color:#454545>The AQI prediction value of the first"
        f" {length} hours of data\n<color: #d43535>RMSE: {rmse:.2f}</></>"
    )
    flexitext(0, 1.12, subtitle, va="top", ax=ax)

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position(
        [box.x0, box.y0 + box.height * 0.1,
         box.width, box.height * 0.9]
    )

    # Put a legend below current axis
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.8, 1.15), # original (0.5, -0.05)
        fancybox=True, shadow=True, ncol=5
    )

    plt.xlabel(None)

    if save:
        plt.savefig(
            rf'{img_save_dir}',
            bbox_inches='tight'
        )
