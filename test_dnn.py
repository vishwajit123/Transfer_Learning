from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import Input, Dense,Flatten
from keras.models import Model
from keras.models import model_from_json


import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 20


model_json = open('model.json','r').read()
model = model_from_json(model_json)

model.load_weights("model.h5")
print("Loaded model from disk")


def test(x):
	"""x is of shape (batch_size,512)"""
	out = model.predict(x)
	print(np.argmax(out)) #Class of x
	return out


if __name__ == '__main__':
	test(np.random.random(1,512))
	#Just an example
