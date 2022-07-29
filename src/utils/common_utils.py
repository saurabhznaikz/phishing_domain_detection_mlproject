import os
import shutil
from pathlib import Path

import yaml


def read_params(config_path: str)-> dict:
    """
                        Method Name: read_params
                        Description: This method performs reading parameters from param.yaml and is a helper
                        function for stage_01_data_preprocessing
                        Output: Return all configuration of yaml to all stages of ML pipeline
                        On Failure: Raise Error

                         Written By: Saurabh Naik
                        Version: 1.0
                        Revisions: None

                        """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)

    return config

def clean_prev_dirs_if_exists(dir_path: str):
    """
                        Method Name: clean_prev_dirs_if_exists
                        Description: This method performs removal of directory if it already exists in order to
                        help stage_01_data_preprocessing.
                        Output: Removes the directory of earlier iteration
                        On Failure: Raise Error

                         Written By: Saurabh Naik
                        Version: 1.0
                        Revisions: None

                        """
    p = Path(__file__).parents[2]
    path = str(p)
    dir_path=os.path.join(path, dir_path)
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)

def create_dir(dirs:list):
    """
                        Method Name: create_dir
                        Description: This method performs creation of directory to help stage_01_data_preprocessing
                        to store preprocessed data into it
                        Output: Creates a directory
                        On Failure: Raise Error

                         Written By: Saurabh Naik
                        Version: 1.0
                        Revisions: None

                        """
    for dir_path in dirs:
        p = Path(__file__).parents[2]
        path = str(p)
        os.makedirs(os.path.join(path, dir_path),exist_ok=True)

def correlation(dataset, threshold):
    """
                        Method Name: correlation
                        Description: This method performs finding correlation among all features of input data
                        and then depending upon the threhold return list of features.
                        Output: Return a list of features having correlation greater than the threshold
                        On Failure: Raise Error

                         Written By: Saurabh Naik
                        Version: 1.0
                        Revisions: None

                        """
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr
