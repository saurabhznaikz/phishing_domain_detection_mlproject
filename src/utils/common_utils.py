import os
import shutil
from pathlib import Path

import yaml


def read_params(config_path: str)-> dict:
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)

    return config

def clean_prev_dirs_if_exists(dir_path: str):
    p = Path(__file__).parents[2]
    path = str(p)
    dir_path=os.path.join(path, dir_path)
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)

def save_local_df(df,df_path,header=False):
    if header:
        new_cols=[col.replace(" ","_") for col in df.columns]
        df.to_csv(df_path,index=False,header=new_cols)
    else:
        df.to_csv(df_path, index=False )

def create_dir(dirs:list):
    for dir_path in dirs:
        p = Path(__file__).parents[2]
        path = str(p)
        os.makedirs(os.path.join(path, dir_path),exist_ok=True)

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr
