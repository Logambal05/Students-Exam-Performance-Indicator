import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    # This class defines configuration parameters for data transformation
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    # This class encapsulates the data transformation config process.
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transfomer_object(self):
        # This Function Responsible for data Transformation,This method creates and returns a preprocessor object.
        try:
            Numerical_Features = ['reading_score', 'writing_score']
            Categorical_Features = [
                'gender', 
                'race_ethnicity', 
                'parental_level_of_education', 
                'lunch', 
                'test_preparation_course'
            ]

            # Pipeline for numerical columns
            Numerical_Pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("standardscaler",StandardScaler())
                ]
            )

            # Pipeline for Categorical columns
            Categorical_Pipeline = Pipeline(
                steps=[
                    ("impute",SimpleImputer(strategy="most_frequent")),
                    ("one_hotencoder",OneHotEncoder()),
                    ("standardscaler",StandardScaler(with_mean=False))
                ]
            )

            
            logging.info(f"Categorical columns: {Categorical_Features}")
            logging.info(f"Numerical columns: {Numerical_Features}")

            # ColumnTransformer to apply different pipelines to numerical and categorical columns
            Preprocessor = ColumnTransformer(
                [
                    ("Numerical_Pipeline",Numerical_Pipeline,Numerical_Features),
                    ("Categorical_pipeline",Categorical_Pipeline,Categorical_Features)
                ]
            )

            return Preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    

    def initiate_data_transformation(self,Train_Path,Test_path):
        try:
            # Reading Train and test Data
            Train_Df = pd.read_csv(Train_Path)
            Test_Df = pd.read_csv(Test_path)
         
            logging.info("Read Train and Test Data Completed")
            logging.info("Obtaining PreProcessor Object")

            Preprocessor_Obj = self.get_data_transfomer_object()
            
            # Seprating Target Feature
            Target_Feature_Name = "math_score"
            Numerical_Feature_Name = ["reading_score", "writing_score"]
            
            # Data Prepared For Training
            Input_Feature_Train_df = Train_Df.drop(columns=[Target_Feature_Name],axis=1)
            Target_Feature_Train_df = Train_Df[Target_Feature_Name]

            # Data Prepared For Testing
            Input_Feature_Test_df = Test_Df.drop(columns=[Target_Feature_Name],axis=1)
            Target_Feature_Test_df = Test_Df[Target_Feature_Name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            # Preprocessing The data using fit nd transfrom
            Input_Feature_Train_Arr = Preprocessor_Obj.fit_transform(Input_Feature_Train_df)
            Input_Feature_Test_Arr = Preprocessor_Obj.transform(Input_Feature_Test_df)

            # Concat the target feature and independent feature
            Train_Arr = np.c_[Input_Feature_Train_Arr, np.array(Target_Feature_Train_df)]
            Test_Arr = np.c_[Input_Feature_Test_Arr, np.array(Target_Feature_Test_df)]


            # Saving the Pkl File 
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = Preprocessor_Obj
            )

            logging.info("Saved Preprocessing Object as Pkl File")

            return Train_Arr,Test_Arr, self.data_transformation_config.preprocessor_obj_file_path
            
        
        except Exception as e:
            raise CustomException(e,sys)