"""
module to import and preprocess epa taiwan data
"""
import os
import pathlib
import glob
from typing import List
import pandas as pd
from dateutil import parser
import torch
from torch.utils.data import Dataset, DataLoader

root_dir = pathlib.Path(os.getcwd()).parent
raw_data_dir = root_dir / "data/0_raw"
processed_data_dir = root_dir / "data/1_processed"
experiment_dir = root_dir / "experiment"

def import_epa_data(
    site_name:str = None,
    year:int = None
    ) -> pd.DataFrame:
    """Read the filtered (by year and column name) csv data of epa taiwan air quality metrics

    Args:
        site_name (str, optional): the site to include. Defaults to None.
        year (int, optional): the year to include. Defaults to None.

    Returns:
        pd.DataFrame: the epa taiwan data as pandas dataframe
    """
    # Import the data
    raw_epa_taiwan_dir = raw_data_dir / "epa_taiwan"
    epa_taiwan_csv_list = glob.glob(f'{str(raw_epa_taiwan_dir)}/*.csv')

    if year:
        # read only specific year from the csv according to user input
        selected_year = [item for item in epa_taiwan_csv_list if str(year)[-2:] in item]
        df = pd.read_csv(selected_year[0])
    else:
        # read all year
        df =  pd.concat([pd.read_csv(csv) for csv in epa_taiwan_csv_list])

    if site_name:
        # read only specific site name
        return df[df["SiteEngName"] == site_name]

    # read all sites
    return df

# Function to convert datetime to naive datetime
def convert_to_naive(dt):
    dt_obj = parser.parse(dt)
    return dt_obj.replace(tzinfo=None)

def standardize_df(
    df:pd.DataFrame
    ) -> pd.DataFrame:
    """
    1. simplify the column names (remove non alpha numeric, convert all to lowercase)
    2. convert to datetime format
    3. sort the dataframe based on read_time
    
    Args:
        df (pd.DataFrame): the pd.Dataframe to be standardize

    Returns:
        pd.DataFrame: standardized pd.Dataframe
    """
    # remove non alpha numeric, convert all to lowercase
    col_name_mapping = {_: _.lower() for _ in df.columns}
    df = df.rename(columns=col_name_mapping)

    # convert to datetime format
    df['read_time'] = pd.to_datetime(df['read_time'].apply(convert_to_naive))

    # sort the dataframe based on read_time
    df = df.sort_values(by="read_time")
    return df

def min_max_df_norm(
    df:pd.DataFrame,
    target:str='pm2.5',
    cols:List=['pm2.5', 'amb_temp', 'ch4', 'co', 'nmhc']
    ) -> pd.DataFrame:
    """do a normalization to a dataframe

    Args:
        df (pd.DataFrame): the dataframe to be normalized
        target (str, optional): the target to be predicted later. Defaults to 'pm2.5'.
        cols (List, optional): columns that will be normalized. Defaults to ['pm2.5', 'amb_temp', 'ch4', 'co', 'nmhc'].

    Returns:
        Tuple[pd.DataFrame, float, float]: return the normalized df and min and max value of the target
    """
    normalized_column_names = []
    for column in cols:
        normalized_column_name = column + '_normalized'
        normalized_column_names.append(normalized_column_name)
        df[normalized_column_name] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        # max_column_name = column + '_max'
        # df[max_column_name] = df[column].max()
        # min_column_name = column + '_min'
        # df[min_column_name] = df[column].min()

    return df, normalized_column_names

class AqiDataset(Dataset):
    def __init__(self, data, history_len, col_names, device):
        self.data = data
        self.history_len = history_len
        self.col_names = col_names
        self.device = device

    def __len__(self):
        self.len = len(self.data) - self.history_len
        return self.len

    def __getitem__(self, index):
        x_cols = self.col_names
        y_cols = ['pm2.5_normalized']
        x = self.data.iloc[index: index+self.history_len, :][x_cols].values
        y = self.data.iloc[index+self.history_len, :][y_cols].values.astype('float')
        if self.device:
            x = torch.tensor(x).float().to(self.device)
            y = torch.tensor(y).float().to(self.device)
        else:
            x = torch.tensor(x).float()
            y = torch.tensor(y).float()
        return x, y
