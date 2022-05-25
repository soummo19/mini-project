from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re

# Keras
from keras.models import load_model
from keras_preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

model  = load_model('modelWt.h5')

print('Model loaded. Check http://127.0.0.1:5000/')

# img_path = 'chest_xray/test/PNEUMONIA/person19_virus_50.jpeg'

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))

    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    img_data = preprocess_input(x)
    classes = model.predict(img_data)
    predNormal = classes[0][0]
    predPneumonia = classes[0][1]
    
    verdict = ""
    if predNormal > predPneumonia:
        verdict = "NORMAL"
    else:
        verdict = "PNEUMONIC"
    
    return verdict

# print(model_predict(img_path, model))


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)

        print(basepath)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)
        print(preds)
        return preds
    return None


if __name__ == '__main__':
   app.run(debug=True)

