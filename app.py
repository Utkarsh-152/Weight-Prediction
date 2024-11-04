from flask import Flask,render_template,request
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.Pipeline.prediction_pipeline import CustomData, PredictPipeline   

application = Flask(__name__)
app = application

#Route for home page
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/data_pred', methods= ['GET','POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template('home.html') 
    else:
        data = CustomData(
            Gender=request.form.get('Gender'),
            PhysicalActivityLevel=request.form.get('PhysicalActivityLevel'),
            SleepQuality=request.form.get('SleepQuality'),
            Age= float(request.form.get('Age')),
            BMR= float(request.form.get('BMR')),
            DailyCaloriesConsumed=float(request.form.get('DailyCaloriesConsumed')),
            Duration= float(request.form.get('Duration')),
            CurrentWeight = float(request.form.get('CurrentWeight')),
            StressLevel = float(request.form.get('StressLevel')),
            WeightChange = float(request.form.get('WeightChange')),
            CaloricSurplusOrDeficit = float(request.form.get('CaloricSurplusOrDeficit'))
            )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)