# Larger CNN for the MNIST Dataset
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras import backend as K

import pickle

K.set_image_dim_ordering('tf')
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#load data etc
X = pickle.load(open("features.pickle", "rb"))
Y = pickle.load(open("labels.pickle", "rb"))
print(X.shape[1:])

Y = np_utils.to_categorical(Y)
training_data = X[0:901]
test_data =X[901:]

training_labels = Y[0:901]
test_labels = Y[901:]

#define the larger model
def larger_model():
    #Create model
    model=Sequential()
    model.add(Conv2D(30,(5,5), input_shape=training_data.shape[1:], activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(15,(5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(5,activation='softmax'))
    #Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#build the model
model=larger_model();
#fit the model
model.fit(training_data, training_labels, batch_size=32, epochs=10, validation_split=0.1)
#test the model
scores = model.evaluate(test_data, test_labels, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

predictions = model.predict(test_data)
print("Predictions:\n", predictions[1])
print("Labels:\n", test_labels[1])

# save the model to a json file
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# save the weight to a HDF5 file
model.save_weights("model.h5")
