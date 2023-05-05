import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder



from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
           
            numerical_cols = ['Delivery_person_Age', 'Delivery_person_Ratings','Delivery_location_latitude', 'Delivery_location_longitude',
            'Vehicle_condition', 'multiple_deliveries','Order_Date_year','Order_Date_month','Order_Date_day']
            # Define which columns should be ordinal-encoded and which should be onehot encoded

            #onehot
            categorical_cols1 = [ 'Type_of_order','Time_Orderd','Time_Order_picked']

            #ordinal
            categorical_cols2 = ['Weather_conditions', 'Road_traffic_density','Type_of_vehicle', 'Festival', 'City']
            

            
            logging.info('Pipeline Initiated')
            
            # Define the custom ranking for each ordinal variable
            Weather_conditions_map=['Sunny','Stormy','Sandstorms','Windy','Cloudy','Fog']
            Road_traffic_density_map=['Low','Medium','High','Jam']
            Type_of_vehicle_map=['electric_scooter','scooter','bicycle',"motorcycle"]
            Festival_map=['No','Yes']
            City_map=['Urban','Metropolitian','Semi-Urban']


            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())])

            # Categorigal Pipeline
            cat_pipeline2=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[Weather_conditions_map,Road_traffic_density_map,Type_of_vehicle_map,Festival_map,City_map])),
                    ('scaler',StandardScaler(with_mean=False))
                    ])
            cat_pipeline1=Pipeline(
                steps=[('imputer',SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))])


            

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline1',cat_pipeline1,categorical_cols1),
                ('cat_pipeline2',cat_pipeline2,categorical_cols2)
                ])
                        
            
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)
        
    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)



            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Time_taken (min)'
            target_feature_train_sr=train_df[target_column_name]
            target_feature_train_df=target_feature_train_sr.to_frame()
            target_feature_test_sr=test_df[target_column_name]
            target_feature_test_df=target_feature_test_sr.to_frame()

            drop_columns = [target_column_name]
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            

            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_train_arr=input_feature_train_arr.toarray()
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            input_feature_test_arr=input_feature_test_arr.toarray()
            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]





            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                 train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path)
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)