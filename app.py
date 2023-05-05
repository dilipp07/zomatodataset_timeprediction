from flask import Flask,request,render_template,jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline
from src.exception import CustomException
import os
import sys

application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    try:
        return render_template('index.html')
    except Exception as e:
        raise CustomException(e,sys)

@app.route('/predict',methods=['GET','POST'])


def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            Delivery_person_Age=float(request.form.get('Delivery_person_Age')),
            Delivery_person_Ratings = float(request.form.get('Delivery_person_Ratings')),
            Delivery_location_latitude = float(request.form.get('Delivery_location_latitude')),
            Delivery_location_longitude = float(request.form.get('Delivery_location_longitude')),
            Order_Date = request.form.get('Order_Date'),
            Time_Orderd = request.form.get('Time_Orderd'),
            Time_Order_picked = request.form.get('Time_Order_picked'),
            Weather_conditions= request.form.get('Weather_conditions'),
            Road_traffic_density = request.form.get('Road_traffic_density'),
            Vehicle_condition = float(request.form.get('Vehicle_condition')),
            Type_of_order = request.form.get('Type_of_order'),
            Type_of_vehicle = request.form.get('Type_of_vehicle'),
            multiple_deliveries =float (request.form.get('multiple_deliveries')),
            Festival= request.form.get('Festival'), 
            City = request.form.get('City')
        )
        final_new_data=data.get_data_as_dataframe()
        print(final_new_data)
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('result.html',final_result=results)






if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)