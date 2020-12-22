from __future__ import division, print_function
import streamlit as st
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf
from PIL import Image, ImageOps

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

from werkzeug.utils import secure_filename

MODEL_PATH = 'model121.h5'
model = load_model(MODEL_PATH)
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


st.write("""
         # Tomato Plant disease prediction
         """
         )
st.write("predicting tomato disease using images of leaf of tomato")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])




# Save the file to ./uploads

def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = x / 255
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x)

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
        preds = "The Disease is Tomato_healthy"
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

    return preds

if file is None:
    st.text("Please upload an image file")
else:

    prediction = model_predict(file, model)

    st.write(prediction)





