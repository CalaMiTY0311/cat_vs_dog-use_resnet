import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

data_train,ds_info = tfds.load('cats_vs_dogs',split = [tfds.Split.TRAIN], with_info=True)
#print(ds_info)
images = [one['image'].numpy() for one in data_train[0].take(30)]
#print(len(images))
#print(images)

"""
plt.imshow(images[20])
plt.show()
plt.axis('off')
"""
#-----------------------------RESNET 이라는 ImageNet 대회에서 15년도 우승했던 이미 학습된 ResNet모델불러와서 사용한 것----------------------------------------
resnet50_pre = tf.keras.applications.resnet.ResNet50(weights='imagenet', input_shape=(224,224,3))
#resnet50_pre.summary()

from tensorflow.keras.applications.imagenet_utils import decode_predictions

def pred_img(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    img_resized = cv2.resize(img,(224,224))
    pred = resnet50_pre.predict(img_resized.reshape([1,224,224,3]))
    decoded_pred = decode_predictions(pred)

    for i, instance in enumerate(decoded_pred[0]):
        print('{}위 : {} ({:.2f}%)'.format(i+1, instance[1],instance[2]*100))

pred_img(images[13])