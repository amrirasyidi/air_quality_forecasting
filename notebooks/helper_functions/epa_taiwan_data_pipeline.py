"""
module to import and preprocess epa taiwan data
"""
import glob
import os
import pathlib

import pandas as pd
import torch
from torch.utils.data import Dataset

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
        df = pd.concat([pd.read_csv(csv) for csv in epa_taiwan_csv_list])

    if site_name:
        # read only specific site name
        return df[df["SiteEngName"] == site_name]

    # read all sites
    return df

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
