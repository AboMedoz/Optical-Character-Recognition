import os

import cv2
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from helpers import preprocess_img

BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(ROOT, 'data')
TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train')

imgs = []
labels = []

with open(os.path.join(TRAIN_DATA_PATH, 'train_gt.txt'), 'r') as f:
    for line in f:
        line = line.strip()
        parts = line.split()

        img = cv2.imread(os.path.join(TRAIN_DATA_PATH, parts[0]))
        img = preprocess_img(img, -1)

        imgs.append(img)
        labels.append(" ".join(parts[1:]))

alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789. "
char_to_idx = {c: i+1 for i, c in enumerate(alphabet)}  # +1 to reserve 0 for CTC blank

int_labels = []
label_lengths = []

for word in labels:
    seq = [char_to_idx[c] for c in word if c in char_to_idx]
    int_labels.append(seq)
    label_lengths.append(len(seq))

max_label_len = max(label_lengths)
padded_labels = pad_sequences(int_labels, maxlen=max_label_len, padding='post')

input_lengths = np.ones(len(imgs)) * (128 // 4)

np.savez_compressed(
    os.path.join(DATA_PATH, 'processed', 'processed.npz'),
    images=np.array(imgs, dtype='float32'),
    labels=np.array(padded_labels),
    input_lengths=np.array(input_lengths),
    label_lengths=np.array(label_lengths)
)
