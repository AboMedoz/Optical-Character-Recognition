import os

import numpy as np
import tensorflow as tf

from ctc_loss import CTCLossLayer

# Macros
BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(BASE_DIR)
MODELS_PATH = os.path.join(ROOT, 'models')
PROCESSED_DATA_PATH = os.path.join(ROOT, 'data', 'processed')

BATCH_SIZE = 32

# Data Loading
data = np.load(os.path.join(PROCESSED_DATA_PATH, 'processed.npz'), allow_pickle=True)

imgs = data['images']
y = data['labels']
input_lengths = data['input_lengths']
label_lengths = data['label_lengths']

img_height = imgs.shape[1]
max_label_length = y.shape[1]
num_classes = np.max(y) + 1

# mask to filter zero label len
non_empty_mask = label_lengths > 0

imgs = imgs[non_empty_mask]
y = y[non_empty_mask]
input_lengths = input_lengths[non_empty_mask]
label_lengths = label_lengths[non_empty_mask]

y = np.clip(y, 0, num_classes - 2)

calculated_input_lengths = np.ones(len(imgs)) * imgs.shape[2]

print(f"Filtered dataset size: {len(imgs)}")
print(f"Max label length: {np.max(label_lengths)} | Min input length: {np.min(calculated_input_lengths)}")

# Model
input_img = tf.keras.layers.Input(shape=(img_height, None, 1), name="image")
labels = tf.keras.layers.Input(name="labels", shape=(max_label_length,))
input_length = tf.keras.layers.Input(name="input_length", shape=(1,))
label_length = tf.keras.layers.Input(name="label_length", shape=(1,))

# CNN
x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(input_img)
x = tf.keras.layers.MaxPooling2D((2, 1))(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = tf.keras.layers.MaxPooling2D((2, 1))(x)
x = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
x = tf.keras.layers.MaxPooling2D((2, 1))(x)
x = tf.keras.layers.Permute((2, 1, 3))(x)
x = tf.keras.layers.Reshape((-1, x.shape[2]*x.shape[3]))(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
y_pred = tf.keras.layers.Dense(num_classes, activation="softmax", name="y_pred")(x)
# Custom CTC loss layer
ctc_loss = CTCLossLayer()([labels, y_pred, input_length, label_length])
model = tf.keras.Model(
    inputs=[input_img, labels, input_length, label_length],
    outputs=ctc_loss
)
model.compile(optimizer="adam", loss=lambda y_true, y_pred: y_pred)

dataset = tf.data.Dataset.from_tensor_slices(({
    "image": imgs,
    "labels": y,
    "input_length": calculated_input_lengths,
    "label_length": label_lengths
}, np.zeros(len(imgs))))

dataset = dataset.batch(BATCH_SIZE).shuffle(1000).repeat()

# Train
model.fit(dataset, steps_per_epoch=len(imgs) // BATCH_SIZE, epochs=10)

# Save
model.save(os.path.join(MODELS_PATH, 'ocr_model.keras'))
