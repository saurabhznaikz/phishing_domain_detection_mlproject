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
        train_data_path = config["preprocessed_data"]["train_data"]
        test_data_path = config["preprocessed_data"]["test_data"]
        target_col=config["base"]["target_col"]
        random_state = config["base"]["random_state"]
        init = config["elbow_method_parameters"]["init"]
        title = config["elbow_method_parameters"]["title"]
        xlabel = config["elbow_method_parameters"]["xlabel"]
        ylabel = config["elbow_method_parameters"]["ylabel"]
        figure_name_train = config["elbow_method_parameters"]["figure_name_train"]
        figure_name_test = config["elbow_method_parameters"]["figure_name_test"]
        curve = config["elbow_method_parameters"]["curve"]
        direction = config["elbow_method_parameters"]["direction"]
        Cluster = config["elbow_method_parameters"]["Cluster"]
        clustered_dir = config["elbow_method_parameters"]["clustered_dir"]
        clustered_train_data = config["elbow_method_parameters"]["clustered_train_data"]
        clustered_test_data = config["elbow_method_parameters"]["clustered_test_data"]

     # Getting dataframe from the csv after preprocessing
        p = Path(__file__).parents[2]
        train_path = str(p) + str(train_data_path)
        logger.log(file, 'Finding the number of cluster in train data')
        train_data = pd.read_csv(train_path)
        test_path = str(p) + str(test_data_path)
        logger.log(file, 'Finding the number of cluster in test data ')
        test_data = pd.read_csv(test_path)

    # Getting no of clusters in train data from K-means
        X_train = train_data.drop(labels=target_col, axis=1)
        Y_train = train_data[[target_col]]
        number_of_clusters_train = elbow_plot(X_train,random_state,init,title,xlabel,ylabel,figure_name_train,curve,direction)
        logger.log(file, 'The optimum number of clusters in training data is: ' + str(number_of_clusters_train))

    # Getting no of clusters in test data from K-means
        X_test = test_data.drop(labels=target_col, axis=1)
        Y_test = test_data[[target_col]]
        number_of_clusters_test = elbow_plot(X_test, random_state, init, title, xlabel, ylabel, figure_name_test, curve,
                                              direction)
        logger.log(file, 'The optimum number of clusters in test data is: ' + str(number_of_clusters_test))

    # Creating clusters in train data
        kmeans = KMeans(n_clusters=number_of_clusters_train, init=init, random_state=random_state)
        y_kmeans_train = kmeans.fit_predict(X_train)  # divide data into clusters

    # Creating clusters in test data
        kmeans = KMeans(n_clusters=number_of_clusters_test, init=init, random_state=random_state)
        y_kmeans_test = kmeans.fit_predict(X_test)  # divide data into clusters

    #Saving the clustering model of train data
        logger.log(file, 'Starting to Save the train clustering model')
        save_model(kmeans, 'KMeans_train')
        logger.log(file, 'train Model Saved')

    # Saving the clustering model of test data
        logger.log(file, 'Starting to Save the test clustering model')
        save_model(kmeans, 'KMeans_test')
        logger.log(file, 'test Model Saved')

    #Adding Cluster no and target column in training dataframe
        X_train[Cluster] = y_kmeans_train
        logger.log(file, 'Added Cluster column in training Dataframe')
        X_train[target_col] = Y_train
        logger.log(file, "Added Target Column in training dataframe")

    # Adding Cluster no and target column in test dataframe
        X_test[Cluster] = y_kmeans_test
        logger.log(file, 'Added Cluster column in test Dataframe')
        X_test[target_col] = Y_test
        logger.log(file, "Added Target Column in test Dataframe")

    # Creating a new directory preprocessed inside Data and inserting preprocessed df in csv file
        clean_prev_dirs_if_exists(clustered_dir)
        p = Path(__file__).parents[2]
        path = str(p) + str(clustered_dir)
        os.makedirs(path, exist_ok=True)
        X_train=pd.DataFrame(X_train)
        path = str(p) + str(clustered_train_data)
        X_train.to_csv(path, index=False)
        logger.log(file, "Data Clustering completed in train dataset")
        X_test = pd.DataFrame(X_test)
        path = str(p) + str(clustered_test_data)
        X_test.to_csv(path, index=False)
        logger.log(file, "Data Clustering completed in test dataset")

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
