from typing import List

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
