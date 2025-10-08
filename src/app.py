import os

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from ctc_loss import CTCLossLayer
from helpers import decode_batch_predictions, preprocess_img

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE_DIR)
MODELS_PATH = os.path.join(ROOT, 'models')

full_model = tf.keras.models.load_model(os.path.join(MODELS_PATH, 'ocr_model.keras'),
                                        custom_objects={'CTCLossLayer': CTCLossLayer}, safe_mode=False)
model = tf.keras.Model(
    inputs=full_model.input[0],  # First input is the image
    outputs=full_model.get_layer("y_pred").output
)


char_list = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789. "

st.header("Optical Character Recognition (OCR)")
st.subheader("Upload Image to Extract text from it")

img = st.file_uploader(label='Image', type=['jpg', 'png'])
if img:
    st.image(img)

    img_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    image = preprocess_img(image, -1)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)

    pred = model.predict(image, verbose=0)
    decoded = decode_batch_predictions(pred)
    for p in decoded:
        predicted_chars = [
            char_list[int(c) - 1] for c in p
            if c != -1 and 0 < int(c) <= len(char_list)
        ]
        st.write('## Predcition')
        st.write(''.join(predicted_chars))

