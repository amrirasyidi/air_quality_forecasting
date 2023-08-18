"""
module to import and preprocess epa taiwan data
"""
import os
import pathlib
import glob
from typing import List
import pandas as pd
from dateutil import parser

root_dir = pathlib.Path(os.getcwd()).parent
raw_data_dir = root_dir / "data/0_raw"
processed_data_dir = root_dir / "data/1_processed"
experiment_dir = root_dir / "experiment"

def import_epa_data(
    site_name:str = None,
    year:int = None,
    columns:List = ["SiteEngName","PM2.5","AMB_TEMP","CH4",'CO',"NMHC","read_time"]
    ) -> pd.DataFrame:
    """Read the filtered (by year and column name) csv data of epa taiwan air quality metrics

    Args:
        site_name (str, optional): the site to include. Defaults to None.
        year (int, optional): the year to include. Defaults to None.
        columns (List, optional): the list of columns to include. 
        Defaults to ["SiteEngName","PM2.5","AMB_TEMP","CH4",'CO',"NMHC","read_time"].

    Returns:
        pd.DataFrame: the epa taiwan data as pandas dataframe
    """
    # Import the data
    raw_epa_taiwan_dir = raw_data_dir / "epa_taiwan"
    epa_taiwan_csv_list = glob.glob(f'{str(raw_epa_taiwan_dir)}/*.csv')

    if year:
        # read only specific year from the csv according to user input
        selected_year = [item for item in epa_taiwan_csv_list if str(year)[-2:] in item]
        df = pd.read_csv(selected_year[0])[columns]
    else:
        # read all year
        df =  pd.concat([pd.read_csv(csv) for csv in epa_taiwan_csv_list])[columns]

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
