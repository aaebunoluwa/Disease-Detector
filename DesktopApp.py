import streamlit as st
import tempfile
from tensorflow.keras.models import load_model
import cv2

model = load_model(r"C:\Users\USER\Documents\TripleA\Downloads\MalariaCellImages\cell_images\malaria_model.h5")

#print(model.summary())
st.title('Disease Detector App')
img = st.file_uploader('Upload cell image', type=['png', 'jpg', 'jpeg'])
tmpfile = tempfile.NamedTemporaryFile(delete=False)

if img:
    tmpfile.write(img.read())
    cv_img = cv2.imread(tmpfile.name)
    cv_img = cv2.resize(cv_img, (60, 60))
    st.image(cv_img, channels='BGR', use_column_width=True)

btn_detect = st.button('Detect')

if btn_detect:
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    cv_img = cv_img/255
    cv_img = cv_img.reshape((1,)+cv_img.shape)
    prob = model.predict(cv_img)
    if prob >= 0.5:
        print(1)
        st.text('Malaria parasite detected!!!')
    else:
        print(0)
        st.text('No Malaria parasite detected!!!')


