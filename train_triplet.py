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
    
def triplet_loss_v2(y_true, y_pred):
    positive, negative = y_pred[:,0,0], y_pred[:,1,0]
    margin = K.constant(0.35)
    loss = K.mean(K.maximum(K.constant(0), positive - negative + margin))
    return loss

def euclidean_distance(vects):
    x, y = vects
    dist = K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
    return dist



try:
    with open('bestmodel.txt', 'r') as file:
        best_model_path = file.read()
    FRmodel = keras.models.load_model(best_model_path, custom_objects={'triplet_loss_v2': triplet_loss_v2})
    
except FileNotFoundError:
    print('Saved model not found, loading FaceNet')
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    load_weights_from_FaceNet(FRmodel)

for layer in FRmodel.layers[0: 100]:
    layer.trainable  =  False

# Model Structure
input_shape=(3, 96, 96)
anchor = Input(shape=input_shape, name = 'anchor')
anchorPositive = Input(shape=input_shape, name = 'anchorPositive')
anchorNegative = Input(shape=input_shape, name = 'anchorNegative')

anchorCode = FRmodel(anchor)
anchorPosCode = FRmodel(anchorPositive)
anchorNegCode = FRmodel(anchorNegative)


positive_dist = Lambda(euclidean_distance, name='pos_dist')([anchorCode, anchorPosCode])
negative_dist = Lambda(euclidean_distance, name='neg_dist')([anchorCode, anchorNegCode])
stacked_dists = Lambda(lambda vects: K.stack(vects, axis=1), name='stacked_dists')([positive_dist, negative_dist])




# Callbacks
early_stopping = EarlyStopping(monitor='loss', patience=1, min_delta=0.00005)
STAMP = 'facenet_%d'%(len(paths)) 
checkpoint_dir = './' + 'checkpoints/' + str(int(time.time())) + '/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

bst_model_path = checkpoint_dir + STAMP + '.h5'
tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))


# Model
tripletModel = Model([anchor, anchorPositive, anchorNegative], stacked_dists, name='triple_siamese')
tripletModel.compile(optimizer = 'adadelta', loss = triplet_loss_v2, metrics = ['accuracy'])



gen = batch_generator(64)
tripletModel.fit_generator(gen, epochs=10,steps_per_epoch=30,callbacks=[early_stopping, tensorboard])

gen = batch_generator2(64)
tripletModel.fit_generator(gen, epochs=10,steps_per_epoch=30,callbacks=[early_stopping, tensorboard])
tripletModel.summary()







# Freezing more layers
for layer in FRmodel.layers[0: 130]:
    layer.trainable  =  False
early_stopping = EarlyStopping(monitor='loss', patience=2, min_delta=0.0000001)

gen = batch_generator(64)
tripletModel.compile(optimizer = 'adadelta', loss = triplet_loss_v2, metrics = ['accuracy'])
tripletModel.fit_generator(gen, epochs=10,steps_per_epoch=30,callbacks=[early_stopping, tensorboard])

gen = batch_generator2(64)
tripletModel.fit_generator(gen, epochs=10,steps_per_epoch=30,callbacks=[early_stopping, tensorboard])
tripletModel.summary()

FRmodel.save(bst_model_path)

with open('bestmodel.txt','w') as file:
    file.write(bst_model_path)