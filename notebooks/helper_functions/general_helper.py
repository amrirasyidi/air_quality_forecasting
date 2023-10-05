import json
import os
import pathlib

import mlflow


def initiate_mlflow_dagshub(
    secret_json:pathlib.WindowsPath,
    experiment_name:str
) -> int:
    """Create a new mlflow experiment within dagshub by using credential from the 
    provided file

    Args:
        secret_json (pathlib.WindowsPath): file containing the credentials
        experiment_name (str): name of the experiment

    Returns:
        int: the experiment id
    """
    if secret_json:
        with open(secret_json, "r") as f:
            secrets = json.load(f)

        os.environ['MLFLOW_TRACKING_USERNAME'] = secrets['MLFLOW_TRACKING_USERNAME']
        os.environ['MLFLOW_TRACKING_PASSWORD'] = secrets['MLFLOW_TRACKING_PASSWORD']
        os.environ['MLFLOW_TRACKING_URI'] = secrets['MLFLOW_TRACKING_URI']

        print("environment variable set!")

    # Check if the experiment exists, and if not, create it
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
    return mlflow.get_experiment_by_name(experiment_name).experiment_id
