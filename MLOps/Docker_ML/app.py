from flask import render_template, request, jsonify,json, Flask
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    uploaded_file = request.files['file']
    df = pd.read_csv(uploaded_file)
    with open('data/model.pkl', 'rb') as f:
        classifier = joblib.load(f)
    prediction_test = classifier.predict(df)
    df['Predicted Flower Type'] = prediction_test
    return df.to_json(orient='split')


if __name__ == '__main__':
    app.run(debug=True,port=5002)