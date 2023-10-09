from typing import List

import numpy as np
import pandas as pd
from dateutil import parser


# Function to convert datetime to naive datetime
def convert_to_naive(dt):
    dt_obj = parser.parse(dt)
    return dt_obj.replace(tzinfo=None)

def standardize_df(
    df:pd.DataFrame,
    dt_col_name: str
    ) -> pd.DataFrame:
    """
    1. simplify the column names (remove non alpha numeric, convert all to lowercase)
    2. convert to datetime format
    3. sort the dataframe based on read_time

    Args:
        df (pd.DataFrame): the pd.Dataframe to be standardize
        dt_col_name (str): the column name of datetime column

    Returns:
        pd.DataFrame: standardized pd.Dataframe
    """
    # remove non alpha numeric, convert all to lowercase
    col_name_mapping = {_: _.lower() for _ in df.columns}
    df = df.rename(columns=col_name_mapping)

    # convert to datetime format
    df[dt_col_name] = pd.to_datetime(df[dt_col_name].apply(convert_to_naive))

    # sort the dataframe based on read_time
    df = df.sort_values(by="read_time")
    return df

def min_max_series_norm(series:pd.Series):
    return (series-series.min())/(series.max()-series.min())

def series_norm(series:pd.Series):
    return (series-series.mean())/series.std()

def df_norm(
    df:pd.DataFrame,
    cols:List,
    norm_method:str="min_max"
    ) -> pd.DataFrame:
    """do a normalization to a dataframe

    Args:
        df (pd.DataFrame): the dataframe to be normalized
        cols (List, optional): columns that will be normalized.

    Returns:
        Tuple[pd.DataFrame, float, float]: the normalized df and min and max value of the target
    """
    normalized_column_names = []
    for column in cols:
        normalized_column_name = column + '_normalized'
        normalized_column_names.append(normalized_column_name)
        if norm_method=="min_max":
            df = df.assign(**{normalized_column_name: min_max_series_norm(df[column])})
        else:
            df[normalized_column_name] = series_norm(df[column])
        # max_column_name = column + '_max'
        # df[max_column_name] = df[column].max()
        # min_column_name = column + '_min'
        # df[min_column_name] = df[column].min()

    return df, normalized_column_names

def create_dt_features(
    df: pd.DataFrame,
    target_variable: str,
    dt_col: str
    ):
    """Creates time series features from datetime index

    Args:
        df (pd.DataFrame): the dataframe which contains datetime column to be turned
                            into time series forecasting features and target
        target_variable (str): the target variable name

    Returns:
        X (int): Extracted values from datetime index, dataframe
        y (int): Values of target variable, numpy array of integers
    """
    # df[dt_col] = df.index
    df['hour'] = df[dt_col].dt.hour
    df['dayofweek'] = df[dt_col].dt.dayofweek
    df['quarter'] = df[dt_col].dt.quarter
    df['month'] = df[dt_col].dt.month
    df['year'] = df[dt_col].dt.year
    df['dayofyear'] = df[dt_col].dt.dayofyear
    df['dayofmonth'] = df[dt_col].dt.day
    df['weekofyear'] = df[dt_col].dt.isocalendar().week.astype('int32')

    # X = df[['hour','dayofweek','quarter','month','year',
    #        'dayofyear','dayofmonth','weekofyear']]
    if target_variable:
        y = df[target_variable]
        return df, y
    return df

def encode_dt_cycle(data, col):
    # https://www.kaggle.com/code/avanwyk/encoding-cyclical-features-for-deep-learning
    # https://harrisonpim.com/blog/the-best-way-to-encode-dates-times-and-other-cyclical-features
    max_val = data['col'].max()
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data
