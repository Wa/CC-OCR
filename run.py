from siamese_network import build_siamese_model
import config
import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist
import numpy as np

import cv2
import sys
import os
import PIL
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

import tensorflow as tf
config1 = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config1)


print("[INFO] loading sample dataset ...")
trainpath = 'sample_dataset\\'
words = os.listdir(trainpath)
category_number = len(words)
img_size = (105, 105)

def loadOneWord(order):
    path = trainpath + words[order] + '\\'
    files = os.listdir(path)
    datas = []
    for file in files:
        file = path + file
        img = np.asarray(Image.open(file).convert('L'))
        img = cv2.resize(img, img_size)
        datas.append(img)
    datas = np.array(datas)
    labels = [order] * len(datas)
    return datas, labels

def transData():
    num = len(words)
    datas = np.array([], dtype=np.uint8)
    datas.shape = -1, 105, 105
    labels = np.array([], dtype=np.uint8)
    for k in tqdm(range(num)):
        data, label = loadOneWord(k)
        datas = np.append(datas, data, axis=0)  
        labels = np.append(labels, label, axis=0)
    return datas, labels

(trainX, trainY) = transData()
testX, testY = trainX, trainY

trainX = trainX / 255.0
testX = testX / 255.0

# add a channel dimension to the images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)

# prepare the positive and negative pairs
print("[INFO] preparing positive and negative pairs...")
(pairTrain, labelTrain) = utils.make_pairs(trainX, trainY)
(pairTest, labelTest) = utils.make_pairs(testX, testY)

# configure the siamese network
print("[INFO] building siamese network...")
imgA = Input(shape=config.IMG_SHAPE)
imgB = Input(shape=config.IMG_SHAPE)
featureExtractor = build_siamese_model(config.IMG_SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

# finally, construct the siamese network
ed = utils.EuclideanDistance()
distance = Dense(1, activation="sigmoid")(ed(featsA, featsB))
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)

# compile the model
print("[INFO] compiling model...")
model.compile(loss="binary_crossentropy", optimizer="adam",
	metrics=["accuracy"])

# train the model
print("[INFO] training model...")
history = model.fit(
	[pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
	validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
	batch_size=config.BATCH_SIZE, 
	epochs=config.EPOCHS)

# serialize the model to disk
print("[INFO] saving siamese model...")
model.save(config.MODEL_PATH)

# plot the training history
print("[INFO] plotting training history...")
utils.plot_training(history, config.PLOT_PATH)

import pickle

# save:
f = open('history.pckl', 'wb')
pickle.dump(history.history, f)
f.close()
