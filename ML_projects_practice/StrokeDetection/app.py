from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from nocache import nocache

app = Flask(__name__)
app.jinja_env.auto_reload = False
app.config["TEMPLATES_AUTO_RELOAD"] = False

# from google.colab import drive
# drive.mount('/content/drive')
 
# import reidge regrssor model and standard scaler
RFC_model=pickle.load(open('Finalpickle/rfc.pkl','rb'))
standard_scaler=pickle.load(open('Finalpickle/stc.pkl','rb'))
Label_encoding=pickle.load(open('FinalPickle/le.pkl','rb'))

## route for home page
@app.route('/',methods=['GET'])
@nocache
def indexone():
    return render_template('F:/AI_ML/heart.html')
@app.route('/pr',methods=['GET','POST'])
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
        bmi = float(request.form.get('bmi'))
        smoking_status = request.form.get('smoking_status')

        new_data_scaled = standard_scaler.transform([[Gender,Age,hypertension,heart_disease,ever_married,work_type,residence_type,smoking_status]])+Label_encoding.transform([[avg_glucose_level,bmi]])
        result = RFC_model.predict(new_data_scaled)

        return render_template('F:/AI_ML/heart.html',result=result)
    else:
        return render_template('F:/AI_ML/heart.html')



if __name__=="__main__":
    app.run(host='0.0.0.0')