import os

import cv2
import pytesseract

BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(BASE_DIR)
TEST_DATA_PATH = os.path.join(ROOT, 'data', 'test')

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

strs = []

for img_name in os.listdir(TEST_DATA_PATH):
    img_path = os.path.join(TEST_DATA_PATH, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
    strs.append(pytesseract.image_to_string(img))

print(*strs, sep='\n')