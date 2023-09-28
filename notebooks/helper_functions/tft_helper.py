from typing import List
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

def generate_dataset(
    train_df:pd.DataFrame,
    time_varying_known_reals:List,
    min_prediction_length:int=24//2,
    max_prediction_length:int=24*7,
):
    max_encoder_length = max_prediction_length * 3
    training_cutoff = train_df["time_idx"].max() - max_prediction_length

    training = TimeSeriesDataSet(
        train_df[lambda x: x.time_idx <= training_cutoff],
        time_idx='time_idx',
        target='pm25',
        group_ids=['stasiun'],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=min_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=['stasiun'],
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=['pm25'],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    return training
