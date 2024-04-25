from flask import Flask,render_template,jsonify,request, url_for 
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__,template_folder='F:\AI_ML\Heart_c')

scaler=pickle.load(open('scaler.pkl','rb'))
rfc=pickle.load(open('rfc.pkl','rb'))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict_datapoint',methods=['GET', 'POST'])
def predict_datapoint():
    result=''
    if request.method =='POST':
        result = ''
        trstbps = int(request.form.get("trstbps"))
        chol = float(request.form.get('chol'))
        thalach = float(request.form.get('thalach'))
        oldpeak = float(request.form.get('oldpeak'))
        cp = float(request.form.get('cp'))
        fbs = float(request.form.get('fbs'))
        restecg = float(request.form.get('restecg'))
        exang = float(request.form.get('exang'))
        slope = float(request.form.get('slope'))
        ca = float(request.form.get('ca'))
        thal = float(request.form.get('thal'))
        Gender = float(request.form.get('Gender'))
        Age = int(request.form.get('Age'))

        numerical_data = [[Age, trstbps, chol, thalach, oldpeak]]
        scaled_data = scaler.fit_transform(numerical_data)

        categorical_data = np.array([[Gender, cp, fbs, restecg, exang, slope, ca, thal]])

        final_data = np.concatenate((scaled_data, categorical_data), axis=1)

        prediction = rfc.predict(final_data)

        if prediction[0] == 1:
            result = 'Heart Attack'
        else:
            result = 'No Heart Attack'

    return render_template('home.html', prediction_text=f"Your prediction: {result}")
    # return render_template('home.html', prediction_text=f"Your prediction: {result}")



if __name__=="__main__":
    app.run(host='0.0.0.0',port=5000,debug=True ,threaded=True,use_reloader=False)
