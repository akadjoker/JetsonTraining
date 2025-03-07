import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import glob
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from imgaug import augmenters as iaa

from tensorflow.keras.models import load_model

def preProcess(img):
    img = img[120:480,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    return img

def rotate_steering_wheel(image, steering_angle):
    steering_wheel = cv2.imread('steering_wheel_image.jpg', cv2.IMREAD_UNCHANGED)
    
    steering_wheel = cv2.resize(steering_wheel, (200, 200))
    
    height, width = steering_wheel.shape[:2]
    center = (width // 2, height // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, -steering_angle * 45, 1.0)
    
    rotated_wheel = cv2.warpAffine(steering_wheel, rotation_matrix, (width, height), 
                                   flags=cv2.INTER_LINEAR, 
                                   borderMode=cv2.BORDER_CONSTANT, 
                                   borderValue=(0,0,0,0))
    
    return rotated_wheel


resize_resolution=(640, 480)

cap = cv2.VideoCapture("videos/session_20250227_203034/video_20250227_203204.avi")

if not cap.isOpened():
    print(f"Erro ao abrir o vídeo: cropped_video")
    exit(1)

 
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Vídeo original: {width}x{height}, FPS: {fps}")
print(f"Redimensionando para: {resize_resolution[0]}x{resize_resolution[1]}")

 
frameCounter = 0
tf.keras.config.enable_unsafe_deserialization()
model = load_model('steering_model_final.keras')

paused = False
while cap.isOpened():
    if not paused:
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        ret, frame = cap.read()
        if not ret:
            break
        
        
 
        process = preProcess(frame)
        image = np.array([process])
        steering = float(model.predict(image)[0])
        frame = cv2.resize(frame, resize_resolution)
        
        steering_wheel = rotate_steering_wheel(frame, steering)
        
        # Posicionar volante no canto inferior direito
        frame_height, frame_width = frame.shape[:2]
        wheel_height, wheel_width = steering_wheel.shape[:2]
        
        # do volante
        x_offset = frame_width - wheel_width - 20
        y_offset = frame_height - wheel_height - 20
        
        # Adicionar canal alfa
        for c in range(0, 3):
            frame[y_offset:y_offset+wheel_height, x_offset:x_offset+wheel_width, c] = \
                steering_wheel[:,:,c] * (steering_wheel[:,:,3]/255.0) + \
                frame[y_offset:y_offset+wheel_height, x_offset:x_offset+wheel_width, c] * (1.0 - steering_wheel[:,:,3]/255.0)
        
        cv2.rectangle(frame, (x_offset, y_offset), (x_offset + wheel_width, y_offset + wheel_height), (0, 0, 255), 2)

        cv2.putText(
            frame, 
            f"Frame: {frameCounter}/{frame_count}", 
            (20, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, (0, 255, 255), 2
        )

        cv2.putText(
            frame, 
            f"Steering: {steering}", 
            (20, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, (0, 255, 255), 2
        )
        

        cv2.imshow("Lane Detection", frame)
        cv2.imshow("Process", process)
    
    

    key = cv2.waitKey(1)
    if key == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
