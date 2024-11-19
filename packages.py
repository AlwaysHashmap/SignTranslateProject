import cv2 as cv2                        # openCV
import numpy as np
import os as os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from matplotlib import pyplot as plt
import time as time
import mediapipe as mediapipe
import string
import keyboard
from PIL import ImageFont, ImageDraw, Image

from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from itertools import product
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import openai
from groq import Groq

mediapipe_holistic = mediapipe.solutions.holistic       # MediaPipe Holistic Model (Detecting)
mediapipe_drawing = mediapipe.solutions.drawing_utils   # MediaPipe Drawing Utilities (Drawing)

NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),                             # Thumb connections
    (0, 5), (5, 6), (6, 7), (7, 8),                             # Index finger connections
    (0, 9), (9, 10), (10, 11), (11, 12),                        # Middle finger connections
    (0, 13), (13, 14), (14, 15), (15, 16),                      # Ring finger connections
    (0, 17), (17, 18), (18, 19), (19, 20)                       # Pinky finger connections
]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),                             # Left face connections
    (0, 4), (4, 5), (5, 6), (6, 8),                             # Right face connections
    (9, 10),                                                    # Mouth connections
    (11, 12), (11, 23), (12, 24), (23, 24),                     # Torso connections
    (11, 13), (13, 15), (15, 21), (15, 19), (15, 17), (17, 19), # Left arm connections
    (12, 14), (14, 16), (16, 22), (16, 20), (16, 18), (18, 20), # Right arm connections
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),           # Left leg connections
    (24, 26), (26, 28), (28, 32), (30, 32), (28, 30)            # Right leg connections
]

__all__ = ['cv2', 'np', 'os', 'tf', 'plt', 'time','mediapipe', 'string', 'keyboard', 'mediapipe_holistic', 'mediapipe_drawing', 'load_model',
           'train_test_split', 'to_categorical', 'product', 'metrics', 'Sequential', 'LSTM', 'Dense', 'ImageFont', 'ImageDraw', 'Image', 'openai', 'Groq',
           'NUM_POSE_LANDMARKS', 'NUM_HAND_LANDMARKS', 'HAND_CONNECTIONS', 'POSE_CONNECTIONS']