from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import imutils
from FaceDetector import *
from parameters import *
import pickle
import sys
# =============================================================================
# np.set_printoptions(threshold=np.nan)
# =============================================================================
import keras

best_model_path =""
if(os.path.exists("bestmodel.txt")):
    with open('bestmodel.txt', 'r') as file:
        best_model_path = file.read()
    
with open("./path_dict.p", 'rb') as f:
    paths = pickle.load(f)
    
faces = []
for key in paths.keys():
    paths[key] = paths[key].replace("\\", "/")
    faces.append(key)
    
if(len(faces) == 0):
    print("No images found in database!!")
    print("Please add images to database")
    sys.exit()

def triplet_loss(y_true, y_pred, alpha = ALPHA):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))    
    return loss

if os.path.exists(best_model_path) and best_model_path !="":
    print("Trained model found")
    print("Loading custom trained model...")
    FRmodel = keras.models.load_model(best_model_path,custom_objects={'triplet_loss': triplet_loss})

else:
    print("Custom trained model not found, Loading original facenet model...")
    FRmodel = faceRecoModel(input_shape=(3, IMAGE_SIZE, IMAGE_SIZE))
    load_weights_from_FaceNet(FRmodel)



def verify(image_path, identity, database, model):
    
    encoding = img_to_encoding(image_path, model, False)
    min_dist = 1000
    for  pic in database:
        dist = np.linalg.norm(encoding - pic)
        if dist < min_dist:
            min_dist = dist
    print(identity + ' : ' +str(min_dist)+ ' ' + str(len(database)))
    
    if min_dist<THRESHOLD:
        door_open = True
    else:
        door_open = False
        
    return min_dist, door_open


database = {}
for face in faces:
    database[face] = []

for face in faces:
    for img in os.listdir(paths[face]):
        database[face].append(img_to_encoding(os.path.join(paths[face],img), FRmodel))


camera = cv2.VideoCapture(0)
fd = faceDetector('fd_models/haarcascade_frontalface_default.xml')

fourcc = cv2.VideoWriter_fourcc(*'XVID') #codec for video
out = cv2.VideoWriter('output.avi', fourcc, 20, (800, 600) )#Output object

while True:
    ret, frame = camera.read()
    frame = imutils.resize(frame, width = 800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(frame.shape)
    faceRects = fd.detect(gray)
    for (x, y, w, h) in faceRects:
        roi = frame[y:y+h,x:x+w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi,(IMAGE_SIZE, IMAGE_SIZE))
        min_dist = 1000
        identity = ""
        detected  = False
        
        for face in range(len(faces)):
            person = faces[face]
            dist, detected = verify(roi, person, database[person], FRmodel)
            if detected == True and dist<min_dist:
                min_dist = dist
                identity = person
        if detected == True:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, identity, (x+ (w//2),y-2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)
            
    cv2.imshow('frame', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
out.release()
cv2.destroyAllWindows()
