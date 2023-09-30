from flask import Flask,render_template,jsonify,request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

scaler=pickle.load(open('model/scaler.pkl','rb'))
model=pickle.load(open('model/ridge.pkl','rb'))

# route for home page
@app.route('/')
def index():
    return render_template('F:/AI & ML/flask/LogML/index.html')

## Route for Single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=""

    if request.method=='POST':

        Pregnancies=int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict=model.predict(new_data)
       
        if predict[0] ==1 :
            result = 'Diabetic'
        else:
            result ='Non-Diabetic'
            
        return render_template('F:/AI & ML/flask/LogML/single_prediction.html',result=result)

    else:
        return render_template('F:/AI & ML/flask/LogML/home.html')



if __name__=="__main__":
    app.run(host="0.0.0.0")