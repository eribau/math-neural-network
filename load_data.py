import numpy as np
from PIL import Image
from numpy import *
import os.path

import matplotlib.pyplot as plt
import matplotlib

from sklearn.utils import shuffle

# Get the filepath to the data folder
path_to_file = os.path.abspath(os.path.dirname(__file__))
path_to_data = os.path.join(path_to_file, "../data")

# Load the data
data = os.listdir(path_to_data)
num_samples = size(data)
print(num_samples)

image = array(Image.open(path_to_data + '/' + data[0]))
m,n = image.shape[0:2]
num_images = len(data);

data_matrix = array([array(Image.open(path_to_data + '/' + image)).flatten() for image in data],'f')

label = np.ones((num_samples,),dtype = int)
label[0:200] = 0        # equals
label[200:400] = 1      # minus
label[400:600] = 2      # pi
label[600:800] = 3      # plus
label[800:1000] = 4     # sigma

data,Label = shuffle(data_matrix,label,random_state=2)
train_data=[data,Label]

img = data_matrix[600].reshape(45,45)
plt.imshow(img)
plt.imshow(img,cmap='gray')
plt.show()
print(train_data[0].shape)
print(train_data[1].shape)
