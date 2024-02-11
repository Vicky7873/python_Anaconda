from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# from google.colab import drive
# drive.mount('/content/drive')

# import reidge regrssor model and standard scaler
ridge_model=pickle.load(open('C:/Users/User/pythonWithAnaconda/ML_projects_practice/StrokeDetection/rfc.pkl','rb'))
standard_scaler=pickle.load(open('C:/Users/User/pythonWithAnaconda/ML_projects_practice/StrokeDetection/pre-process.pkl','rb'))

## route for home page
@app.route('/',methods=['GET','POSt'])
def index():
    return render_template('F:/AI_ML/heart.html')

def predict_datapoint():
    if request.method=='POST':
        Gender =request.form.get('Gender')
        Age = float(request.form.get('Age'))
        hypertension = request.form.get('hypertension')
        heart_disease = request.form.get('heart_disease')
        ever_married = request.form.get('ever_married')
        work_type = request.form.get('work_type')
        residence_type = request.form.get('residence_type')
        avg_glucose_level = float(request.form.get('avg_glucose_level'))
        bmi = float(request.form.get('BMI'))
        smoking_status = request.form.get('smoking_status')

        new_data_scaled = standard_scaler.transform([[Gender,Age,hypertension,heart_disease,ever_married,work_type,residence_type,avg_glucose_level,bmi,smoking_status]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('F:/AI_ML/heart.html',result=result[1])
    else:
        return render_template('F:/AI_ML/heart.html')



if __name__=="__main__":
    app.run(host='0.0.0.0')