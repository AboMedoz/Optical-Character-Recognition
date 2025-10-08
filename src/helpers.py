import cv2
import numpy as np
import tensorflow as tf


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    results = results.numpy()
    return results


def preprocess_img(img, dim_axis, img_size=(64, 256)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape

    target_h, target_w = img_size

    scale = target_h / h
    new_w = int(w * scale)

    resized = cv2.resize(img, (new_w, target_h))

    padded = np.zeros((target_h, target_w), dtype=np.uint8)
    padded[:, :min(new_w, target_w)] = resized[:, :min(new_w, target_w)]

    padded = padded.astype('float32') / 255.0
    padded = np.expand_dims(padded, axis=dim_axis)
    return padded


