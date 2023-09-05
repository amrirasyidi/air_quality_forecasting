import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable

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
    fig, axs = plt.subplots(nrow,ncol,figsize = (20, 20), sharey=True)

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
    reg_func,
    shift:int=-1,
    parameter:str="pm2.5",
    ):
    temp_df = data[[parameter]]
    lagged_param_name = parameter+"_lag"
    temp_df[lagged_param_name] = temp_df[parameter].shift(shift)
    x=temp_df[parameter]
    y=temp_df[lagged_param_name]

    fig, ax = plt.subplots()

    slope, intercept, xseq = reg_func(x.iloc[:shift],y[:shift])

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
    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9,
                                 center="light",
                                 as_cmap=True
                                )

    plt.figure(figsize=(16, 12))

    sns.heatmap(matrix, mask=mask, center=0, annot=True,
                fmt='.2f', square=True, cmap=cmap)

    plt.show()
