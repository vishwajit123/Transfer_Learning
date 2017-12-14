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

input_shape = 512


def load_data():
	parent_dir = 'features_npz/'
	sub_dirs = os.listdir(parent_dir)

	x = []
	y = []

	for folder in sub_dirs:
		print(folder)
		for file in os.listdir(parent_dir+folder):
			try:
				a = np.load(parent_dir + folder + '/' + file)
				a = a['arr_0']
				if len(a)==4: #Throw other examples
					x.append(a.flatten())
					label = file.split('-')[1]
					y.append(int(label))
			except:
				pass			

	return np.array(x),np.array(y)
		
x,y = load_data()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)


x_train = x_train / 255.0
x_test = x_test / 255.0

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


inputs = Input(shape=(512,))

layer_1 = Dense(256, activation='relu')(inputs)
layer_2 = Dense(128, activation='relu')(layer_1)
layer_2 = Dropout(0.3)(layer_2)
layer_3 = Dense(128, activation='relu')(layer_2)
layer_3 = Dropout(0.3)(layer_3)
layer_4 = Dense(64, activation='relu')(layer_3)

predictions = Dense(10, activation='softmax')(layer_4)


model = Model(inputs=inputs, outputs=predictions)
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(lr=1.4,decay =0.0001),metrics=['accuracy'])
history = model.fit(x_train, y_train,batch_size=batch_size,validation_split=0.1,epochs=epochs,verbose=1,validation_data=(x_test, y_test))


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('accuracy.png')
plt.clf()

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('loss.png')


model_json = model.to_json()
with open("model.json", "w") as json_file:
		json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
