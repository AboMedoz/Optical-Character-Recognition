import random

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


# Macros
MAX_LABEL_LENGTH = 10   # max word length
BATCH_SIZE = 32
IMG_HEIGHT = 28
MAX_WIDTH = 128


def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def dataset_iterator(dataset):
    # Convert tf.data to numpy generator that repeats forever
    for example in tfds.as_numpy(dataset.repeat().shuffle(10000)):
        yield example


def make_word_image(iterator, max_len=MAX_LABEL_LENGTH):
    """Generate a fake word by stitching EMNIST chars horizontally"""
    word_len = random.randint(3, max_len)
    chars, labels = [], []

    for _ in range(word_len):
        img, lbl = next(iterator)   # use persistent iterator
        img = img.squeeze()            # 28x28
        chars.append(img)
        labels.append(int(lbl))

    word_img = np.hstack(chars)  # concatenate horizontally
    max_width = 28 * max_len

    padded = np.ones((IMG_HEIGHT, max_width)) * 255
    padded[:, :word_img.shape[1]] = word_img

    return padded.astype("float32") / 255.0, labels


def data_generator(iterator, batch_size=BATCH_SIZE, max_len=MAX_LABEL_LENGTH):
    while True:
        batch_imgs, batch_labels, input_lengths, label_lengths = [], [], [], []

        for _ in range(batch_size):
            img, lbls = make_word_image(iterator, max_len)
            img = np.expand_dims(img, -1)  # add channel
            batch_imgs.append(img)

            label_lengths.append(len(lbls))
            lbls = lbls + [0] * (max_len - len(lbls))  # pad
            batch_labels.append(lbls)

            width = img.shape[1]
            seq_len = width // 4   # after pooling stride
            input_lengths.append(seq_len)

        inputs = {
            "image": np.array(batch_imgs),
            "labels": np.array(batch_labels),
            "input_length": np.array(input_lengths),
            "label_length": np.array(label_lengths),
        }
        outputs = np.zeros(batch_size)  # dummy target for CTC
        yield inputs, outputs


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    results = results.numpy()
    return results


def preprocess_img(img, dim_axis, img_size=(64, 28)):
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


