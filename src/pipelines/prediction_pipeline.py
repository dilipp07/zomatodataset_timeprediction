import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
import numpy as np

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            logging.info("done prediction")
            return pred
            logging.info("returned pred value")

            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)




class CustomData:
    def __init__(self,
    Delivery_person_Age:float,
    Delivery_person_Ratings:float,
    Delivery_location_latitude:float,
    Delivery_location_longitude:float,
    
    Time_Orderd:str,
    Time_Order_picked:str,
    Weather_conditions:str,
    Road_traffic_density:str,
    Vehicle_condition:float,
    Type_of_order:str,
    Type_of_vehicle:str,
    multiple_deliveries:float,
    Festival:str,

    City:str,
    Order_Date_year:float,
    Order_Date_month:float,
    Order_Date_day:float):
    
        
        self.Delivery_person_Age=Delivery_person_Age,
        self.Delivery_person_Ratings=Delivery_person_Ratings,
        self.Delivery_location_latitude=Delivery_location_latitude,
        self.Delivery_location_longitude=Delivery_location_longitude,
        
        self.Time_Orderd=Time_Orderd,
        self.Time_Order_picked=Time_Order_picked,
        self.Weather_conditions=Weather_conditions,
        self.Road_traffic_density=Road_traffic_density,
        self.Vehicle_condition=Vehicle_condition,
        self.Type_of_order=Type_of_order,
        self.Type_of_vehicle=Type_of_vehicle,
        self.multiple_deliveries=multiple_deliveries,
        self.Festival=Festival,
        self.City=City,
        self.Order_Date_year=Order_Date_year,
        self.Order_Date_month=Order_Date_month,
        self.Order_Date_day=Order_Date_day,
        

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Delivery_person_Age':[self.Delivery_person_Age][0],
                'Delivery_person_Ratings':[self.Delivery_person_Ratings][0],
                'Delivery_location_latitude':[self.Delivery_location_latitude][0],
                'Delivery_location_longitude':[self.Delivery_location_longitude][0],
                
                'Time_Orderd':[self.Time_Orderd][0],
                'Time_Order_picked':[self.Time_Order_picked][0],
                'Weather_conditions':[self.Weather_conditions][0],
                'Road_traffic_density':[self.Road_traffic_density][0],
                'Vehicle_condition':[self.Vehicle_condition][0],
                'Type_of_order':[self.Type_of_order][0],
                'Type_of_vehicle':[self.Type_of_vehicle][0],
                'multiple_deliveries':[self.multiple_deliveries][0],
                'Festival':[self.Festival][0],
                'City':[self.City][0],
                'Order_Date_year':[self.Order_Date_year][0],
                'Order_Date_month':[self.Order_Date_month][0],
                'Order_Date_day':[self.Order_Date_day][0],
                }

            df = pd.DataFrame(custom_data_input_dict)
           
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)