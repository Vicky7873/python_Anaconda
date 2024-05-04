from __future__ import division, print_function
from flask import render_template,Flask, redirect, url_for, request
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image #type: ignore
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions # type: ignore

# coding=utf-8
import sys
import os
import glob
import re
from werkzeug.utils import secure_filename
from PIL import Image # type: ignore

app=Flask(__name__)
model=load_model('model_resnet50.h5')


@app.route('/')
def index():
    return render_template('index.html')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

   

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The Car IS Audi"
    elif preds==1:
        preds="The Car is Lamborghini"
    else:
        preds="The Car Is Mercedes"
    
    
    return preds

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None

if __name__== "__main__":
    app.run(host="0.0.0.0",debug=True)
