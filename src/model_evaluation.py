import os

import cv2
import numpy as np
import tensorflow as tf

from ctc_loss import CTCLossLayer
from helpers import decode_batch_predictions, preprocess_img

# Macros
BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(BASE_DIR)
MODELS_PATH = os.path.join(ROOT, 'models')
TEST_DATA_PATH = os.path.join(ROOT, 'data', 'test')

# load model
full_model = tf.keras.models.load_model(os.path.join(MODELS_PATH, 'ocr_model.keras'),
                                   custom_objects={'CTCLossLayer': CTCLossLayer}, safe_mode=False)

model = tf.keras.Model(
    inputs=full_model.input[0],  # First input is the image
    outputs=full_model.get_layer("y_pred").output
)

predictions = []

for img_name in os.listdir(TEST_DATA_PATH):
    img_path = os.path.join(TEST_DATA_PATH, img_name)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = preprocess_img(img)

    pred = model.predict(img, verbose=0)

    decoded = decode_batch_predictions(pred)
    for p in decoded:
        predicted_chars = [int(c) for c in p if c != -1]
        predictions.append(predicted_chars)

print(predictions)
