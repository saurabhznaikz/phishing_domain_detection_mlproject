import os
import shutil
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle

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
        print(os.path.join(path, dir_path))
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

def elbow_plot(data,random_state,init,title,xlabel,ylabel,figure_name,curve,direction):
    """
                        Method Name: elbow_plot
                        Description: This method saves the plot to decide the optimum number of clusters to the file.
                        Output: A picture saved to the directory
                        On Failure: Raise Exception

                        Written By: Saurabh Naik
                        Version: 1.0
                        Revisions: None

    """
    wcss=[] # initializing an empty list
    try:
        for i in range (1,11):
            kmeans=KMeans(n_clusters=i,init=init,random_state=random_state) # initializing the KMeans object
            kmeans.fit(data) # fitting the data to the KMeans Algorithm
            wcss.append(kmeans.inertia_)
        plt.plot(range(1,11),wcss) # creating the graph between WCSS and the number of clusters
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.show()
        plt.savefig(figure_name) # saving the elbow plot locally
        # finding the value of the optimum cluster programmatically
        kn = KneeLocator(range(1, 11), wcss, curve=curve, direction=direction)
        return kn.knee
    except Exception as e:
        raise e

def save_model(model,filename):
        """
                                Method Name: save_model
                                Description: Save the model file to directory
                                Outcome: File gets saved
                                On Failure: Raise Exception

                                Written By: Saurabh Naik
                                Version: 1.0
                                Revisions: None
        """
        model_directory = 'models/'
        try:
            path = os.path.join(model_directory,filename) #create seperate directory for each cluster
            if os.path.isdir(path): #remove previously existing models for each clusters
                shutil.rmtree(model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path) #
            with open(path +'/' + filename+'.sav',
                      'wb') as f:
                pickle.dump(model, f) # save the model to file

        except Exception as e:
            raise e