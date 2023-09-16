"""functions to help with xgboost pipeline
"""
import warnings
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure

# from flexitext import flexitext
from sklearn import metrics

warnings.filterwarnings("ignore")

def create_features(df, target_variable):
    """
    Creates time series features from datetime index
    
    Args:
        df (float64): Values to be added to the model incl. corresponding datetime
                      , numpy array of floats
        target_variable (string): Name of the target variable within df   
    
    Returns:
        X (int): Extracted values from datetime index, dataframe
        y (int): Values of target variable, numpy array of integers
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype('int32')

    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if target_variable:
        y = df[target_variable]
        return X, y
    return X

def mean_absolute_percentage_error_func(y_true, y_pred):
    '''
    Calculate the mean absolute percentage error as a metric for evaluation
    
    Args:
        y_true (float64): Y values for the dependent variable (test part), numpy array of floats 
        y_pred (float64): Predicted values for the dependen variable (test parrt), numpy array of floats
    
    Returns:
        Mean absolute percentage error 
    '''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def timeseries_evaluation_metrics_func(y_true, y_pred):
    '''
    Calculate the following evaluation metrics:
        - MSE
        - MAE
        - RMSE
        - MAPE
        - R²
    
    Args:
        y_true (float64): Y values for the dependent variable (test part), numpy array of floats 
        y_pred (float64): Predicted values for the dependen variable (test parrt), numpy array of floats
    
    Returns:
        MSE, MAE, RMSE, MAPE and R² 
    '''
    #print('Evaluation metric results: ')
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MAPE is : {mean_absolute_percentage_error_func(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}',end='\n\n')

def xgb_time_train_test_split(
    all_data:pd.DataFrame,
    train_frac:float,
    val_frac:float,
    test_frac:float
):
    """_summary_

    Args:
        all_data (pd.DataFrame): _description_
        train_frac (float): _description_
        val_frac (float): _description_
        test_frac (float): _description_

    Returns:
        _type_: _description_
    """
    if test_frac:
        assert train_frac+val_frac+test_frac==1, \
            "train_frac+val_frac+test_frac must equal to 1"
        # Calculate the split indices
        total_len = len(all_data)
        train_len = int(total_len * train_frac)
        val_len = int(total_len * val_frac)

        # Split the DataFrame into train, validation, and test sets
        df_train = all_data[:train_len]
        df_val = all_data[train_len:train_len + val_len]
        df_test = all_data[train_len + val_len:]

        print(
            f"length of train: {df_train.shape[0]} "
            f"| {round((df_train.shape[0]/total_len)*100)}% of total length\n"
            f"length of validation: {df_val.shape[0]} "
            f"| {round((df_val.shape[0]/total_len)*100)}% of total length\n"
            f"length of test: {df_test.shape[0]} "
            f"| {round((df_test.shape[0]/total_len)*100)}% of total length"
        )
        return df_train, df_val, df_test
    else:
        assert train_frac+val_frac==1, \
            "train_frac+val_frac must equal to 1"

        # Calculate the split indices
        total_len = len(all_data)
        train_len = int(total_len * train_frac)
        val_len = int(total_len * val_frac)

        # Split the DataFrame into train, validation, and test sets
        df_train = all_data[:train_len]
        df_val = all_data[train_len:]

        print(
            f"length of train: {df_train.shape[0]} "
            f"| {round((df_train.shape[0]/total_len)*100)}% of total length\n"
            f"length of validation: {df_val.shape[0]} "
            f"| {round((df_val.shape[0]/total_len)*100)}% of total length\n"
        )
        return df_train, df_val

def create_delayed_df(
    original_df:pd.DataFrame,
    period:int
):
    """_summary_

    Args:
        original_df (pd.DataFrame): _description_
        period (int): _description_

    Returns:
        _type_: _description_
    """
    # create delayed df
    pm_minus_period = original_df.copy()
    pm_minus_period = (
        pm_minus_period.assign(lagged_value=pm_minus_period['lin_pm25'].shift(period))
        )
    return pm_minus_period.dropna()

def xgb_predict_plot(
    true_value:pd.DataFrame,
    pred_real:pd.DataFrame,
) -> Tuple[Figure, Axes, pd.DataFrame]:
    """draw prediction as a matplotlib lineplot

    Args:
        pred_real (pd.DataFrame): dataframe with real predicted value
        value_col_name (str): the column name of the predicted value

    Returns:
        Tuple[Figure, Axes, pd.DataFrame]: The plot figure and axes and the validation
                                            real value
    """
    fig, ax = plt.subplots(figsize = (15,6))

    true_value.plot(
        x='read_time',
        y='lagged_value',
        ax=ax,
        color='blue',
        label='True Value'
    )
    
    pred_real_df = pd.DataFrame(
        data=pred_real, 
        index=true_value.index, 
        columns=['Predicted']
    )
    pred_real_df.plot(
        ax=ax,
        color='red',
        style='--',
        label='xgb predicted'
    )

    plt.legend()

    return fig, ax
