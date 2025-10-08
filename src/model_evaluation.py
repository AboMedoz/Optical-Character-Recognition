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

char_list = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789. "

# load model
full_model = tf.keras.models.load_model(os.path.join(MODELS_PATH, 'ocr_model.keras'),
                                   custom_objects={'CTCLossLayer': CTCLossLayer}, safe_mode=False)

model = tf.keras.Model(
    inputs=full_model.input[0],  # First input is the image
    outputs=full_model.get_layer("y_pred").output
)

gt = []
predictions = []

for img_name in os.listdir(TEST_DATA_PATH):
    img_path = os.path.join(TEST_DATA_PATH, img_name)

    img_name = img_name[:-4]
    gt.append(img_name)

    img = cv2.imread(img_path)
    img = preprocess_img(img, -1)

    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)  # add channel
    img = np.expand_dims(img, axis=0)  # add batch dim

    pred = model.predict(img, verbose=0)

    decoded = decode_batch_predictions(pred)
    for p in decoded:
        predicted_chars = [
            char_list[int(c) - 1] for c in p
            if c != -1 and int(c) > 0 and int(c) <= len(char_list)
        ]
        predictions.append(''.join(predicted_chars))

print(f"Ground Truth: {gt}")
print(f"OCR: {predictions}")
