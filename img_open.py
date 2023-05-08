import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
from PIL import Image
import os
import numpy as np

data_train,ds_info = tfds.load('cats_vs_dogs',split = [tfds.Split.TRAIN], with_info=True)
#print(ds_info)

images = [one['image'].numpy() for one in data_train[0].take(30)]

if not os.path.exists('img'):
    os.makedirs('img')

for i in range(len(images)):
    img = Image.fromarray(images[i])
    filename = 'img/img{}.jpg'.format(i+1)
    img.save(filename, 'JPEG')
    print("Saved", filename)


home_img = Image.open('img.jpg')
home_img = np.array(home_img)
images.append(home_img)


