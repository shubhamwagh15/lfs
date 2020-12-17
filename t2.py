
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model121.h5'

# Load your trained model
model = load_model(MODEL_PATH)


#Set Max size of file as 10MB.

#Allow files with extension png, jpg and jpeg
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



# Function to load and prepare the image in right shape


@app.route("/", methods=['GET', 'POST'])
def home():

    return render_template('home.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        try:
            if file and allowed_file(file.filename):
                filename = file.filename
                file_path = os.path.join(filename)
                file.save(file_path)

                # Predict the class of an image
                img = image.load_img(file_path, target_size=(224, 224))

                # Preprocessing the image
                x = image.img_to_array(img)
                # x = np.true_divide(x, 255)
                ## Scaling
                x = x / 255
                x = np.expand_dims(x, axis=0)


                #Map apparel category with the numerical class
                preds = model.predict(x)
                preds = np.argmax(preds, axis=1)
                if preds == 0:
                    preds = "The Disease is Tomato__Bacterial_spot"
                elif preds == 1:
                    preds = "The Disease is Tomato__Early_blight"
                elif preds == 2:
                    preds = "The Disease is Tomato___healthy"
                elif preds == 3:
                    preds = "The Disease is Tomato_Late_blight"
                elif preds == 4:
                    preds = "The Disease is Tomato___Leaf_Mold"
                elif preds == 5:
                    preds = "The Disease is Tomato__Septoria_leaf_spot"
                elif preds == 6:
                    preds = "The Disease is Tomato___Spider_mites Two-spotted_spider_mite"
                elif preds == 7:
                    preds = "The Disease is Tomato_Target_spot"
                elif preds == 8:
                    preds = "The Disease is Tomato_Mosaic_virus"
                elif preds == 9:
                    preds = "The Disease is Tomato___Tomato_Yellow_Leaf_Curl_Virus"
                elif preds == 10:
                    preds = "The Disease is Pepper__bell___Bacterial_spot"
                elif preds == 11:
                    preds = "The Disease is Pepper__bell___Bacterial_spot"
                elif preds == 12:
                    preds = "The Disease is Pepper__bell___Bacterial_spot"
                elif preds == 13:
                    preds = "The Disease is Pepper__bell___Bacterial_spot"

                return render_template('predict.html', prediction = preds )
        except Exception as e:
            return "Unable to read the file. Please check if the file extension is correct."

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(port=5001, debug=True)
