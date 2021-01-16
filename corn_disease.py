import tensorflow as tf
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps

model = tf.keras.models.load_model('mobile_net_transfer_learning.hdf5')


st.write("Chimwe Chi Model")

file = st.file_uploader("Please upload an image file",type = ["jpg","png"])


def import_and_predict(image_data, model):
    image = ImageOps.fit(image_data, (224,224), Image.ANTIALIAS)
    img_array = tf.keras.preprocessing.image.img_to_array(image,)
    img_array = img_array/255       
    img_reshape = np.expand_dims(img_array,axis=0)
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is  Blight!")
    elif np.argmax(prediction) == 1:
        st.write("It is  Common Rust!")
    elif np.argmax(prediction) == 2:
        st.write("It is  Gray Leaf Spot!")
    else:
        st.write("It is  Healthy!")
    
    st.text("Probability (0: Blight, 1: Common Rust, 2: Gray Leaf Spot, 3: Healthy")
    st.write(prediction)