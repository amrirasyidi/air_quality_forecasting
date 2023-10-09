"""functions to help with xgboost training pipeline
"""
from typing import Dict, List

import numpy as np

# import optuna
import pandas as pd
import xgboost as xgb

# from optuna.integration.mlflow import MLflowCallback
from sklearn import metrics


def xgb_train(
    train_feature_df:pd.DataFrame,
    train_target_series: pd.Series,
    val_feature_df:pd.DataFrame,
    val_target_series: pd.Series,
    param:Dict,
    num_round:int,
    patience:int=20,
    # pruning_callbacks:List
    ):
    """train an xgboost model

    Args:
        train_feature_df (pd.DataFrame): training feature dataframe
        train_target_series (pd.Series): training target series
        val_feature_df (pd.DataFrame): validation feature dataframe
        val_target_series (pd.Series): validation target series
        param (Dict): parameters for the xgboost model
        num_round (int): xgboost trainng epoch number
        pruning_callbacks (List): pruning method for the xgboost model

    Returns:
        _type_: _description_
    """
    # Convert the datasets into DMatrix
    dtrain = xgb.DMatrix(train_feature_df, label=train_target_series)
    dval = xgb.DMatrix(val_feature_df, label=val_target_series)

    # Train the model
    # if pruning_callbacks:
    #     xgb_model = xgb.train(param, dtrain, num_round, callbacks=pruning_callbacks)
    # else:
    evals_result = {}
    xgb_model = xgb.train(
        param, dtrain, num_round,
        evals=[(dtrain, 'train'), (dval, 'valid')],
        early_stopping_rounds=patience,
        verbose_eval=False,
        evals_result=evals_result
        )

    # Predict the target for the training set
    train_pred = xgb_model.predict(dtrain)

    # Predict the target for the val set
    val_pred = xgb_model.predict(dval)

    # y_pred_real = lagged_df.lagged_value * np.exp(pred_df.Predicted)
    train_rmse = np.sqrt(metrics.mean_squared_error(train_target_series, train_pred))
    val_rmse = np.sqrt(metrics.mean_squared_error(val_target_series, val_pred))
    return xgb_model, train_pred, train_rmse, val_pred, val_rmse, evals_result

def xgb_predict(
    model:xgb.core.Booster,
    feature_df:pd.DataFrame,
) -> np.ndarray:
    """predict using xgboost model on dmatrix data format

    Args:
        model (xgb.core.Booster): the xgboost model
        feature_df (pd.DataFrame): feature as dataframe

    Returns:
        np.ndarray: prediction result
    """
    # Convert the datasets into DMatrix
    data = xgb.DMatrix(feature_df)
    # Predict the target for the val set
    return model.predict(data)
    