import os

import tensorflow as tf
import tensorflow_datasets as tfds

from ctc_loss import CTCLossLayer
from helpers import data_generator, dataset_iterator, decode_batch_predictions, normalize

# Macros
IMG_HEIGHT = 28
MAX_LABEL_LENGTH = 10   # max word length
NUM_CLASSES = 47 + 1    # 47 chars + 1 blank for CTC
BATCH_SIZE = 32

BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(BASE_DIR)
MODELS_PATH = os.path.join(ROOT, 'models')

emnist = tfds.load("emnist/balanced", split="train", as_supervised=True)
emnist_iter = dataset_iterator(emnist)

# Model
input_img = tf.keras.layers.Input(shape=(IMG_HEIGHT, None, 1), name="image")
labels = tf.keras.layers.Input(name="labels", shape=(MAX_LABEL_LENGTH,))
input_length = tf.keras.layers.Input(name="input_length", shape=(1,))
label_length = tf.keras.layers.Input(name="label_length", shape=(1,))
# CNN
x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(input_img)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
x = tf.keras.layers.MaxPooling2D((2, 1))(x)
x = tf.keras.layers.Permute((2, 1, 3))(x)
x = tf.keras.layers.Reshape((-1, x.shape[2]*x.shape[3]))(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
y_pred = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="y_pred")(x)
# Custom CTC loss layer
ctc_loss = CTCLossLayer()([labels, y_pred, input_length, label_length])
model = tf.keras.Model(
    inputs=[input_img, labels, input_length, label_length],
    outputs=ctc_loss
)

# Train
model.compile(optimizer="adam", loss=lambda y_true, y_pred: y_pred)
train_gen = data_generator(emnist_iter, batch_size=BATCH_SIZE)
model.fit(train_gen, steps_per_epoch=100, epochs=10)

# Save
model.save(os.path.join(MODELS_PATH, 'ocr_model.keras'))

