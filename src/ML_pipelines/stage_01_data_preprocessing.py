from pathlib import Path
import argparse
from src.utils.common_utils import read_params, clean_prev_dirs_if_exists, create_dir,correlation
import pandas as pd
from src.application_logging.logger import App_Logger
from imblearn.over_sampling import SMOTE


def data_preprocessing(config_path):
    """
                    Method Name: data_preprocessing
                    Description: This method performs data preprocessing by reading parameters from param.yaml and then
                    pereforming feature engineering and feature selection based on EDA given in Jupyter notebook
                    notebooks/EDA and preprocessing.
                    Output: Return a preprocessed csv having the data ready for ML algos
                    On Failure: Raise Error

                     Written By: Saurabh Naik
                    Version: 1.0
                    Revisions: None

                    """
    try:

        # Initializing Logger object
        logger = App_Logger()
        p = Path(__file__).parents[2]
        path=str(p)+"/src/Training_Logs/DataPreprocessingLog.txt"
        file = open(path, "a+")
        logger.log(file, "Data preprocessing started ")

        # Reading of params from params.yaml file
        config = read_params(config_path)
        data_path = config["data_source"]["data_source"]
        preprocessed_dir = config["preprocessed_data"]["preprocessed_dir"]
        preprocessed_data = config["preprocessed_data"]["preprocessed_data"]
        target_col=config["base"]["target_col"]
        random_state = config["base"]["random_state"]
        sampling_strategy = config["base"]["sampling_strategy"]

        #Getting dataframe from the csv provided by Database
        p = Path(__file__).parents[2]
        path = str(p)+str(data_path)
        df = pd.read_csv(path)

        #Feature Engineering
        logger.log(file, "Feature Engineering Started...")

        #Feature Engineering: Handling imbalanced dataset.
        logger.log(file, "Handling imbalanced dataset.")
        sm = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        # Fit the model to generate the data.
        oversampled_X, oversampled_Y = sm.fit_resample(df.drop(target_col, axis=1), df[target_col])
        df = pd.concat([pd.DataFrame(oversampled_Y), pd.DataFrame(oversampled_X)], axis=1)
        logger.log(file, "Imbalanced Dataset handled by SMOTE")

        #Feature Engineering: Handling outliers in all features dataset.
        logger.log(file, "Handling outliers in all features of  dataset.")
        for feature in df.columns:
            IQR = df[feature].quantile(0.75) - df[feature].quantile(0.25)
            lower_bridge = df[feature].quantile(0.25) - (IQR * 1.5)
            upper_bridge = df[feature].quantile(0.75) + (IQR * 1.5)
            df.loc[df[feature] < lower_bridge, feature] = lower_bridge
            df.loc[df[feature] >= upper_bridge, feature] = upper_bridge
        logger.log(file, "Outliers have been handled")

        # Feature Selection
        logger.log(file, "Feature Selection Started...")

        #Feature selection:Finding correlated features and removing those features which are 85% correlated
        X = df.drop(labels=target_col, axis=1)
        Y = df[[target_col]]
        corr_features = correlation(X, 0.85)

        #Removing correlated features and then creating a new Dataframe
        X.drop(corr_features, axis=1, inplace=True)
        df_final = pd.DataFrame(X)
        df_final[target_col] = pd.DataFrame(Y)
        logger.log(file, "Feature Selection Completed")

        #Creating a new directory preprocessed inside Data and inserting preprocessed df in csv file
        clean_prev_dirs_if_exists(preprocessed_dir)
        create_dir(dirs=[preprocessed_dir])
        p = Path(__file__).parents[2]
        path = str(p) + str(preprocessed_data)
        df_final.to_csv(path, index=False)
        logger.log(file, "Data preprocessing completed")

    except Exception as e:
        logger = App_Logger()
        p = Path(__file__).parents[2]
        path = str(p) + "/src/Training_Logs/DataPreprocessingLog.txt"
        file = open(path, "a+")
        logger.log(file, "error encountered due to: %s" %e)
        raise e


if __name__ == '__main__':
    p = Path(__file__).parents[2]
    path=str(p) + "\params.yaml"
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=path)
    parsed_args = args.parse_args()

    try:

        data = data_preprocessing(config_path=parsed_args.config)
    except Exception as e:
        raise e
