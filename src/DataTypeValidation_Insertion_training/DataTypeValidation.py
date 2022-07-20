import shutil
from datetime import datetime
import pandas as pd
from os import listdir
import os
import csv
from src.application_logging.logger import App_Logger
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider


class dBOperation:
    """
      This class shall be used for handling all the SQL operations.

      Written By: Saurabh Naik
      Version: 1.0
      Revisions: None

      """
    def __init__(self):
        # self.path = "Training_Database/"
        # self.badFilePath = "Training_Raw_files_validated/Bad_Raw"
        self.goodFilePath = "../data/good_raw"
        self.logger = App_Logger()


    def dataBaseConnection(self):

        """
                Method Name: dataBaseConnection
                Description: This method creates the database with the given name and if Database already exists then opens the connection to the DB.
                Output: Connection to the DB
                On Failure: Raise ConnectionError

                 Written By: Saurabh Naik
                Version: 1.0
                Revisions: None

                """
        try:
            cloud_config = {
                "secure_connect_bundle": "secure-connect-phishing-detector-training.zip"
            }
            auth_provider = PlainTextAuthProvider("hnPokMRfIQTEQTYNXJTFLmKT", "T4Zhc-TLCvUDIsnpXsURD60QDncwTtk5envBeFMtQma3rSI66JxiX5FHCPZv_ukgRQ7j6WUaYK4ZNEZ0S26DRmLlON5_Bmb+B_P5-klAbzIFX_RGNpHbznZKm9gwz9Gg")
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
            session = cluster.connect()
            file = open("Training_Logs/DataBaseConnectionLog.txt", "a+")
            self.logger.log(file, "Opened database connection successfully")
            file.close()
        except ConnectionError:
            file = open("Training_Logs/DataBaseConnectionLog.txt", "a+")
            self.logger.log(file, "Error while connecting to database: %s" %ConnectionError)
            file.close()
            raise ConnectionError
        return session

    def createTableDb(self,column_names):
        """
                        Method Name: createTableDb
                        Description: This method creates a table in the given database which will be used to insert the Good data after raw data validation.
                        Output: None
                        On Failure: Raise Exception

                         Written By: Saurabh Naik
                        Version: 1.0
                        Revisions: None

                        """
        try:
            session = self.dataBaseConnection()
            row = session.execute("SELECT * FROM phishing_training.Good_Raw_Data where id=1")
            length=len(row.column_names)
            if length > 0:
                file = open("Training_Logs/DbTableCreateLog.txt", "a+")
                self.logger.log(file, "Tables created successfully!!")
                file.close()

                file = open("Training_Logs/DataBaseConnectionLog.txt", "a+")
                self.logger.log(file, "Closed database successfully")
                file.close()

            else:
                for key in column_names.keys():
                    type = column_names[key]

                    #in try block we create table and then add columns in it
                    # else in catch block we will find the exception
                    try:
                        # row = session.execute("SELECT * FROM phishing_training.Good_Raw_Data")
                        # session.execute("CREATE TABLE  phishing_training.Good_Raw_Data (id int PRIMARY KEY)")
                        session.execute('ALTER TABLE phishing_training.Good_Raw_Data ADD "{column_name}"{dataType}'.format(column_name=key,dataType=type))
                    except Exception as e:
                        file = open("Training_Logs/DbTableCreateLog.txt", "a+")
                        self.logger.log(file, "Error while creating table: %s " % e)
                        file.close()


                file = open("Training_Logs/DbTableCreateLog.txt", "a+")
                self.logger.log(file, "Tables created successfully!!")
                file.close()

                file = open("Training_Logs/DataBaseConnectionLog.txt", "a+")
                self.logger.log(file, "Closed database successfully")
                file.close()

        except Exception as e:
            file = open("Training_Logs/DbTableCreateLog.txt", "a+")
            self.logger.log(file, "Error while creating table: %s " % e)
            file.close()
            file = open("Training_Logs/DataBaseConnectionLog.txt", "a+")
            self.logger.log(file, "Closed database successfully")
            file.close()
            raise e


    def insertIntoTableGoodData(self):

        """
                               Method Name: insertIntoTableGoodData
                               Description: This method inserts the Good data files from the Good_Raw folder into the
                                            above created table.
                               Output: None
                               On Failure: Raise Exception

                                Written By: Saurabh Naik
                               Version: 1.0
                               Revisions: None

        """

        session = self.dataBaseConnection()
        goodFilePath= self.goodFilePath
        onlyfiles = [f for f in listdir(goodFilePath)]
        log_file = open("Training_Logs/DbInsertLog.txt", "a+")

        for file in onlyfiles:
            try:
                with open(goodFilePath+"/"+file, "r") as f:
                    df1 = pd.read_csv(f)
                    filename="../data/good_raw/raw_data.csv"
                    for i in range(len(df1.index)):
                        df1 = pd.read_csv(filename)
                        if (len(df1.index) != 0):
                            l1 = df1.loc[0].values.tolist()
                            list_ = ",".join(list(map(lambda x: "{y}".format(y=x), l1)))
                            count = session.execute(
                                "SELECT *  FROM phishing_training.Good_Raw_Data where id=1;")
                            if(len(count.current_rows)==0):
                                id=1
                                print("in if block")
                            else:
                                id = session.execute(
                                    "SELECT max(id) as id  FROM phishing_training.Good_Raw_Data").one().id + 1
                            session.execute(
                                'INSERT INTO phishing_training.Good_Raw_Data ("id","asn_ip", "directory_length", "email_in_url", "file_length", "length_url", "params_length", "phishing", "qty_and_directory", "qty_and_file", "qty_and_params", "qty_and_url", "qty_asterisk_directory", "qty_asterisk_file", "qty_asterisk_params", "qty_at_directory", "qty_at_file", "qty_at_params", "qty_at_url", "qty_comma_directory", "qty_comma_file", "qty_comma_params", "qty_dollar_directory", "qty_dollar_file", "qty_dollar_params", "qty_dot_directory", "qty_dot_domain", "qty_dot_file", "qty_dot_params", "qty_dot_url", "qty_equal_directory", "qty_equal_file", "qty_equal_params", "qty_equal_url", "qty_exclamation_directory", "qty_exclamation_file", "qty_exclamation_params", "qty_hashtag_directory", "qty_hashtag_file", "qty_hashtag_params", "qty_hyphen_directory", "qty_hyphen_file", "qty_hyphen_params", "qty_hyphen_url", "qty_ip_resolved", "qty_params", "qty_percent_directory", "qty_percent_file", "qty_percent_params", "qty_plus_directory", "qty_plus_file", "qty_plus_params", "qty_questionmark_directory", "qty_questionmark_file", "qty_questionmark_params", "qty_slash_directory", "qty_slash_file", "qty_slash_params", "qty_slash_url", "qty_space_directory", "qty_space_file", "qty_space_params", "qty_tilde_directory", "qty_tilde_file", "qty_tilde_params", "qty_tld_url", "qty_underline_directory", "qty_underline_file", "qty_underline_params", "qty_underline_url", "qty_vowels_domain", "time_domain_activation", "time_domain_expiration", "tld_present_params", "ttl_hostname") VALUES ({id},{values})'.format(
                                    id=id, values=(list_)))
                            print('Entries remaining :{remaining} Entries Entered : {id}'.format(remaining=len(df1.index),id=id))
                            self.logger.log(log_file, " %s: Entry loaded successfully!!" % file)
                            df1 = df1.iloc[1:]
                            df1.to_csv(filename, index=False)
                        else:
                            break

            except Exception as e:
                self.logger.log(log_file,"Error while inserting in table due to: %s " % e)
                self.logger.log(log_file, "Failed to insert into the database %s" % file)
                log_file.close()

        log_file.close()


    def selectingDatafromtableintocsv(self,Database):

        """
                               Method Name: selectingDatafromtableintocsv
                               Description: This method exports the data in GoodData table as a CSV file. in a given location.
                                            above created .
                               Output: None
                               On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                               Version: 1.0
                               Revisions: None

        """

        self.fileFromDb = "Training_FileFromDB/"
        self.fileName = "InputFile.csv"
        log_file = open("Training_Logs/ExportToCsv.txt", "a+")
        try:
            conn = self.dataBaseConnection(Database)
            sqlSelect = "SELECT *  FROM Good_Raw_Data"
            cursor = conn.cursor()

            cursor.execute(sqlSelect)

            results = cursor.fetchall()
            # Get the headers of the csv file
            headers = [i[0] for i in cursor.description]

            #Make the CSV ouput directory
            if not os.path.isdir(self.fileFromDb):
                os.makedirs(self.fileFromDb)

            # Open CSV file for writing.
            csvFile = csv.writer(open(self.fileFromDb + self.fileName, "w", newline=""),delimiter=",", lineterminator="\r\n",quoting=csv.QUOTE_ALL, escapechar="\\")

            # Add the headers and data to the CSV file.
            csvFile.writerow(headers)
            csvFile.writerows(results)

            self.logger.log(log_file, "File exported successfully!!!")
            log_file.close()

        except Exception as e:
            self.logger.log(log_file, "File exporting failed. Error : %s" %e)
            log_file.close()





