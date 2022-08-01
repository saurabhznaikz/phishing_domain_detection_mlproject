from pathlib import Path
import argparse
import os
from src.application_logging.logger import App_Logger
from src.utils.common_utils import read_params, elbow_plot, save_model, clean_prev_dirs_if_exists
import pandas as pd
from sklearn.cluster import KMeans


def create_clusters(config_path):
    """
                    Method Name: create_clusters
                    Description: Create a new dataframe consisting of the cluster information.
                    Output: A datframe with cluster column
                    On Failure: Raise Exception

                     Written By: Saurabh Naik
                    Version: 1.0
                    Revisions: None

                    """
    try:

    # Initializing Logger object
        logger = App_Logger()
        p = Path(__file__).parents[2]
        path=str(p)+"/src/Training_Logs/DataClusteringLog.txt"
        file = open(path, "a+")
        logger.log(file, "Data Clustering started ")

    # Reading of params from params.yaml file
        config = read_params(config_path)
        preprocessed_data = config["preprocessed_data"]["preprocessed_data"]
        target_col=config["base"]["target_col"]
        random_state = config["base"]["random_state"]
        init = config["elbow_method_parameters"]["init"]
        title = config["elbow_method_parameters"]["title"]
        xlabel = config["elbow_method_parameters"]["xlabel"]
        ylabel = config["elbow_method_parameters"]["ylabel"]
        figure_name = config["elbow_method_parameters"]["figure_name"]
        curve = config["elbow_method_parameters"]["curve"]
        direction = config["elbow_method_parameters"]["direction"]
        Cluster = config["elbow_method_parameters"]["Cluster"]
        clustered_dir = config["elbow_method_parameters"]["clustered_dir"]
        clustered_data = config["elbow_method_parameters"]["clustered_data"]

     # Getting dataframe from the csv provided by Database
        p = Path(__file__).parents[2]
        path = str(p) + str(preprocessed_data)
        logger.log(file, 'Finding the number of cluster')
        df = pd.read_csv(path)

    # Getting no of clusters from K-means
        X = df.drop(labels=target_col, axis=1)
        Y = df[[target_col]]
        number_of_clusters = elbow_plot(X,random_state,init,title,xlabel,ylabel,figure_name,curve,direction)
        logger.log(file, 'The optimum number of clusters is: ' + str(number_of_clusters))

    # Creating clusters
        kmeans = KMeans(n_clusters=number_of_clusters, init=init, random_state=random_state)
        y_kmeans = kmeans.fit_predict(X)  # divide data into clusters

    #Saving the clustering
        logger.log(file, 'Starting to Save the clustering model')
        save_model(kmeans, 'KMeans')
        logger.log(file, 'Model Saved')

    #Adding Cluster no and target column in dataframe
        X[Cluster] = y_kmeans
        logger.log(file, 'Added Cluster column in Dataframe')
        X[target_col] = Y
        logger.log(file, "Added Target Column")

    # Creating a new directory preprocessed inside Data and inserting preprocessed df in csv file
        clean_prev_dirs_if_exists(clustered_dir)
        p = Path(__file__).parents[2]
        path = str(p) + str(clustered_dir)
        os.makedirs(path, exist_ok=True)
        X=pd.DataFrame(X)
        path = str(p) + str(clustered_data)
        X.to_csv(path, index=False)
        logger.log(file, "Data Clustering completed")

    except Exception as e:
        logger = App_Logger()
        p = Path(__file__).parents[2]
        path = str(p) + "/src/Training_Logs/DataClusteringLog.txt"
        file = open(path, "a+")
        logger.log(file, "error encountered due to: %s" % e)
        raise e


if __name__ == '__main__':
    p = Path(__file__).parents[2]
    path=str(p) + "\params.yaml"
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=path)
    parsed_args = args.parse_args()

    try:

        data = create_clusters(config_path=parsed_args.config)
    except Exception as e:
        raise e
