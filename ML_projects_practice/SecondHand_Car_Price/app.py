from flask import Flask,render_template,jsonify,request, url_for # type: ignore
import pickle
import numpy as np # type: ignore
import pandas as pd # type: ignore

app =Flask(__name__,template_folder="F:\AI_ML\second_HC")
pipe_model=pickle.load(open('predict_pipeline.pkl','rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_datapoint",methods=["GET","POST"])
def predict_datapoint():
    result=0
    if request.method == "POST":
        brand=request.form["brand"]
        model=request.form["model"]
        year=request.form["year"]
        fuel_type=request.form["fuel_type"]
        Transmission=request.form["Transmission"]
        Owner_Type=request.form["Owner_Type"]
        Seats=request.form["Seats"]
        kilometers_driven=float(request.form["kilometers"])
        mileage=float(request.form["mileage"])
        engine=float(request.form["engine"])
        power=float(request.form["power"])

        
       # Create input list
        input_data = [[brand, model, year,kilometers_driven,fuel_type,Transmission,Owner_Type,mileage,engine,power,Seats]]  # Add other form fields as needed

        # Define column names for the DataFrame
        columns = ['Brand', 'Model', 'Year','Kilometers_Driven','Fuel_Type','Transmission','Owner_Type','Mileage','Engine','Power','Seats']  # Add other column names as needed

        # Convert input list to DataFrame
        input_df = pd.DataFrame(input_data, columns=columns)

        # Make prediction using the pipeline
        result = pipe_model.predict(input_df)
       
    return render_template("home_shc.html",prediction_text=result)


if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)