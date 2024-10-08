#!/usr/bin/env python
# coding: utf-8

# In[1]:

import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import logging

# Set up logging
log_file_path = os.path.join("C:\\Users\\Navdeep\\Emotion_Detection", "emotion_detection.log")
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classifier = load_model("model.h5")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

# Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow messages

# Example logging inside the prediction loop
while True:
    _, frame = cap.read(0)
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float32') / 255.0  # Normalize the pixel values
            roi = np.expand_dims(roi, axis=-1)  # Add channel dimension (grayscale image)
            roi = np.expand_dims(roi, axis=0)  # Add batch dimension

            prediction = classifier.predict(roi)[0]  # Make prediction
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Log the prediction result
            logging.info(f'Predicted Emotion: {label}')
            print(f'Predicted Emotion: {label}')  # Print to terminal

        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




# In[ ]:




