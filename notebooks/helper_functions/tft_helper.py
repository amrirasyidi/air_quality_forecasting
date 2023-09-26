import pandas as pd


def create_features(
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


