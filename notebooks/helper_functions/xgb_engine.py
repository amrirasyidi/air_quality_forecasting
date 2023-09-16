from typing import Dict

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import metrics

import optuna
from optuna.integration.mlflow import MLflowCallback

def xgb_train(
    train_feature_df:pd.DataFrame,
    train_target_series: pd.Series,
    val_feature_df:pd.DataFrame,
    val_target_series: pd.Series,
    param:Dict,
    num_round:int,
    pruning_callback
    ):
    # Convert the datasets into DMatrix
    dtrain = xgb.DMatrix(train_feature_df, label=train_target_series)
    dval = xgb.DMatrix(val_feature_df, label=val_target_series)

    # Train the model
    if pruning_callback:
        xgb_model = xgb.train(param, dtrain, num_round, callbacks=[pruning_callback])
    else:
        evals_result = {}
        xgb_model = xgb.train(
            param, dtrain, num_round, 
            evals=[(dtrain, 'train'), (dval, 'valid')],
            early_stopping_rounds=20,
            verbose_eval=False,
            evals_result=evals_result
            )

    # xgb_model = xgb.train(param, dtrain, num_round)
    
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
    