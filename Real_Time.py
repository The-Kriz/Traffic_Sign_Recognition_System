import numpy as np
import pandas as pd
import os
import random
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score

classes = [ 'Speed limit (20km/h)',
            'Speed limit (30km/h)',
            'Speed limit (50km/h)',
            'Speed limit (60km/h)',
            'Speed limit (70km/h)',
            'Speed limit (80km/h)',
            'End of speed limit (80km/h)',
            'Speed limit (100km/h)',
            'Speed limit (120km/h)',
            'No passing',
            'No passing veh over 3.5 tons',
            'Right-of-way at intersection',
            'Priority road',
            'Yield',
            'Stop',
            'No vehicles',
            'Veh > 3.5 tons prohibited',
            'No entry',
            'General caution',
            'Dangerous curve left',
            'Dangerous curve right',
            'Double curve',
            'Bumpy road',
            'Slippery road',
            'Road narrows on the right',
            'Road work',
            'Traffic signals',
            'Pedestrians',
            'Children crossing',
            'Bicycles crossing',
            'Beware of ice/snow',
            'Wild animals crossing',
            'End speed + passing limits',
            'Turn right ahead',
            'Turn left ahead',
            'Ahead only',
            'Go straight or right',
            'Go straight or left',
            'Keep right',
            'Keep left',
            'Roundabout mandatory',
            'End of no passing',
            'End no passing veh > 3.5 tons' ]

cam = cv2.VideoCapture(0)
model_path = "model_krizz_4.h5"
loaded_model = tf.keras.models.load_model(model_path)


# define a video capture object
cam = cv2.VideoCapture(0)

while (True):

    # Capture the video frame
    # by frame
    res, image = cam.read()
    image_fromarray = Image.fromarray(image, 'RGB')
    resize_image = image_fromarray.resize((30, 30))
    expand_input = np.expand_dims(resize_image, axis=0)
    input_data = np.array(expand_input)
    input_data = input_data / 255

    pred = loaded_model.predict(input_data)
    result = pred.argmax()
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(image, str(classes[result]), (0, 50), font, 1, (0, 0, 0),2)
    cv2.imshow('Real Time', image)
    print(result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
# Destroy all the windows
cam.destroyAllWindows()