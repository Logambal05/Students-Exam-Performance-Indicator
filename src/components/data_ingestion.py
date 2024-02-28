import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import(
    DataTransformationConfig,
    DataTransformation)
# from src.components.model_training import ModelTrainerConfig,ModelTrainer

@dataclass
class DataIngestionConfig:
    Train_Data_Path: str =  os.path.join("artifacts","train.csv")
    Test_Data_Path: str =  os.path.join("artifacts","test.csv")
    Raw_Data_Path: str =  os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self):
        # Initialize DataIngestionConfig to get configuration parameters
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion Method/Component")
        try:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv("notebook/data/StudentsPerformance.csv")
            logging.info("Read The Dataset as DataFrame")

            # Create directories as per the configuration for storing processed data
            os.makedirs(os.path.dirname(self.ingestion_config.Train_Data_Path), exist_ok=True)

            # Save the entire DataFrame to a CSV file as raw data
            df.to_csv(self.ingestion_config.Raw_Data_Path, index=False, header=True)
            logging.info("Train Test Initiated Using Raw Data")

            # Split the DataFrame into training and testing sets
            Train_Set, Test_Set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the training and Test set to a CSV file
            Train_Set.to_csv(self.ingestion_config.Train_Data_Path, index=False, header=True)
            Test_Set.to_csv(self.ingestion_config.Test_Data_Path, index=False, header=True)

            logging.info("Ingestion Of The Data Completed")
            return (
                self.ingestion_config.Train_Data_Path,
                self.ingestion_config.Test_Data_Path
                )
        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    # Data Ingestion.
    obj = DataIngestion()
    Train_Data,Test_Data=obj.initiate_data_ingestion()

    # Data Transformation.
    Data_Trans = DataTransformation()
    Data_Trans.initiate_data_transformation(Train_Data,Test_Data)

    # Model training
    # Model_Trained = ModelTrainer()
    # print(Model_Trained.Intiate_Model_Trainer(Train_Arr,Test_Arr))