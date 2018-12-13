import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

path_to_file = os.path.abspath(os.path.dirname(__file__))
path_to_data = os.path.join(path_to_file, "../")
categories = ["equals", "minus", "pi", "plus", "sigma"]

img_size = 45


def create_training_data():
    training_data = []
    for category in categories:
        path = os.path.join(path_to_data, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([img_array, class_num])
            except Exception as e:
                pass
    return training_data

training_data = create_training_data()

random.shuffle(training_data)

X = []
Y = []

# Put the data and the labels into their own arrays
for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 1)
X = X/255.0

pickle_out = open("features.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("labels.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()
