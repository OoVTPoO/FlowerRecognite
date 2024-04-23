import os
import tensorflow as tf
from keras.models import load_model
import streamlit as st
import numpy as np

st.title('Tìm hoa bằng hình ảnh')
flower_names = ['Hoa cúc', 'Hoa Bồ Công Anh', 'Hoa Hồng', 'Hoa Hướng Dẫn', 'Hoa Tulip']

model = load_model('my_model.keras')

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array,0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'Hình ảnh này là ' + flower_names[np.argmax(result)] + ' với tỷ lệ là ' + str(round(np.max(result)*100, 2)) + '%'
    return outcome

uploaded_file = st.file_uploader('Tải hình ảnh lên tại đây')
if uploaded_file is not None:
    with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, caption='Hình ảnh đã tải lên', use_column_width=True)

    result = classify_images(uploaded_file)
    st.write(result)