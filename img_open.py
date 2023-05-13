import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from PIL import Image
import os
import numpy as np
from pathlib import Path

data_train,ds_info = tfds.load('cats_vs_dogs',split = [tfds.Split.TRAIN], with_info=True)
#print(ds_info)

tfds_images = [one['image'].numpy() for one in data_train[0].take(30)]
print(tfds_images[0])
def img_save(images):                                   #img save 함수
    for i in range(len(images)):
        img = Image.fromarray(images[i])
        filename = 'img/img{}.jpg'.format(i+1)
        img.save(filename, 'JPEG')
        print("Saved", filename)

if not os.path.exists('img'):
    os.makedirs('img')
    img_save(tfds_images)



def load_images_tonumpy():
    img_dir = "img/"
    # 이미지 파일 목록 가져오기
    img_files = os.listdir(img_dir)

    img_size = (224, 224)
    # 이미지 파일 목록 순회하며 NumPy 배열로 변환하기
    img_arrays = []
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        img = Image.open(img_path).resize(img_size)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_arrays.append(img_array)

    # 변환된 NumPy 배열들을 하나의 배열로 합치기
    img_array = list(np.stack(img_arrays))

    return img_array










