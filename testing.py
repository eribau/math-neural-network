# Larger CNN for the MNIST Dataset
import numpy as np
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
from keras.models import model_from_json
import matplotlib.pyplot as plt
from load_data2 import create_training_data

import pickle

categories = ["equals", "minus", "pi", "plus", "sigma"]

# load an image of a written sigma and display it
img = plt.imread('../plus/plus-100.jpg')
show = plt.imshow(img)
plt.show(show)

example = np.array(img).reshape(-1, 45, 45, 1) # reshape
example = example / 255 # normalize

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Use the model to predict the test image
predictions = loaded_model.predict(example)

result = list(np.around(predictions[0][0:5]))
print("The CNN predicted the image to be a " + categories[result.index(1)])
