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
from keras.models import model_from_json
import keras
from generator_utils import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import time
from parameters import *

def triplet_loss(y_true, y_pred, alpha = ALPHA):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss


best_model_path=None
if os.path.exists("./bestmodel.txt"):
    with open('bestmodel.txt', 'r') as file:
        best_model_path = file.read()
    
if best_model_path != None and os.path.exists(best_model_path):
    ("Pre trained model found")
    FRmodel = keras.models.load_model(best_model_path, custom_objects={'triplet_loss': triplet_loss})
    
else:
    print('Saved model not found, loading untrained FaceNet')
    FRmodel = faceRecoModel(input_shape=(3, IMAGE_SIZE, IMAGE_SIZE))
    load_weights_from_FaceNet(FRmodel)

for layer in FRmodel.layers[0: LAYERS_TO_FREEZE]:
    layer.trainable  =  False
    
    
input_shape=(3, IMAGE_SIZE, IMAGE_SIZE)
A = Input(shape=input_shape, name = 'anchor')
P = Input(shape=input_shape, name = 'anchorPositive')
N = Input(shape=input_shape, name = 'anchorNegative')

enc_A = FRmodel(A)
enc_P = FRmodel(P)
enc_N = FRmodel(N)


# Callbacks
early_stopping = EarlyStopping(monitor='loss', patience=5, min_delta=0.00005)
STAMP = 'facenet_%d'%(len(paths)) 
checkpoint_dir = './' + 'checkpoints/' + str(int(time.time())) + '/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

bst_model_path = checkpoint_dir + STAMP + '.h5'
tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

# Model
tripletModel = Model(input=[A, P, N], output=[enc_A, enc_P, enc_N])
tripletModel.compile(optimizer = 'adam', loss = triplet_loss)

gen = batch_generator(BATCH_SIZE)
tripletModel.fit_generator(gen, epochs=NUM_EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,callbacks=[early_stopping, tensorboard])

FRmodel.save(bst_model_path)
with open('bestmodel.txt','w') as file:
    file.write(bst_model_path)
