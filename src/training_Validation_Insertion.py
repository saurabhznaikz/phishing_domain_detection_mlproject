import os
from datetime import datetime

from src.DataTransform_Training.DataTransformation import dataTransform
from src.Training_Raw_data_validation.rawValidation import Raw_Data_validation
# from DataTypeValidation_Insertion_Training.DataTypeValidation import dBOperation
# from DataTransform_Training.DataTransformation import dataTransform
from src.application_logging import logger
from os import listdir


#
class train_validation:
    def __init__(self, path):
        self.raw_data = Raw_Data_validation(path)
        self.dataTransform = dataTransform()
        # self.dBOperation = dBOperation()
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()

    def train_validation(self):
        try:
            self.log_writer.log(self.file_object, 'Start of Validation on files!!')
            # extracting values from prediction schema
            column_names, noofcolumns = self.raw_data.valuesFromSchema()
            # validating column length in the file
            self.raw_data.validateColumnLength(noofcolumns)
            # validating if any column has all values missing
            self.raw_data.validateMissingValuesInWholeColumn()
            self.log_writer.log(self.file_object, "Raw Data Validation Complete!!")

            self.log_writer.log(self.file_object, "Starting Data Transforamtion!!")
            # below function adds quotes to the '?' values in some columns.
            self.dataTransform.addQuotesToStringValuesInColumn()
            # replacing blanks in the csv file with "Null" values to insert in table
            self.dataTransform.replaceMissingWithNull()

            self.log_writer.log(self.file_object, "DataTransformation Completed!!!")

            # self.log_writer.log(self.file_object,
            #                     "Creating Training_Database and tables on the basis of given schema!!!")
            # # create database with given name, if present open the connection! Create table with columns given in schema
            # self.dBOperation.createTableDb('Training', column_names)
            # self.log_writer.log(self.file_object, "Table creation Completed!!")
            # self.log_writer.log(self.file_object, "Insertion of Data into Table started!!!!")
            # # insert csv files in the table
            # self.dBOperation.insertIntoTableGoodData('Training')
            # self.log_writer.log(self.file_object, "Insertion in Table completed!!!")
            # self.log_writer.log(self.file_object, "Deleting Good Data Folder!!!")
            # # Delete the good data folder after loading files in table
            # self.raw_data.deleteExistingGoodDataTrainingFolder()
            # self.log_writer.log(self.file_object, "Good_Data folder deleted!!!")
            # self.log_writer.log(self.file_object, "Moving bad files to Archive and deleting Bad_Data folder!!!")
            # # Move the bad files to archive folder
            # self.raw_data.moveBadFilesToArchiveBad()
            # self.log_writer.log(self.file_object, "Bad files moved to archive!! Bad folder Deleted!!")
            # self.log_writer.log(self.file_object, "Validation Operation completed!!")
            # self.log_writer.log(self.file_object, "Extracting csv file from table")
            # # export data in table to csvfile
            # self.dBOperation.selectingDatafromtableintocsv('Training')
            # self.file_object.close()

        except Exception as e:
            raise e

if __name__ == '__main__':
    path=os.path.join("data/", "raw/")
    train_valObj = train_validation(path)  # object initialization
    train_valObj.train_validation()  # calling the training_validation function
