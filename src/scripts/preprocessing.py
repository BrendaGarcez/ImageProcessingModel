import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))  # Tamanho compatível com YOLO
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img