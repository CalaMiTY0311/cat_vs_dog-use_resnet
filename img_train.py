import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import os

resnet50_pre = tf.keras.applications.resnet.ResNet50(weights='imagenet', input_shape=(224,224,3))
#resnet50_pre.summary()

from tensorflow.keras.applications.imagenet_utils import decode_predictions
from img_open import images

def pred_img(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    img_resized = cv2.resize(img,(224,224))
    pred = resnet50_pre.predict(img_resized.reshape([1,224,224,3]))
    decoded_pred = decode_predictions(pred)

    for i, instance in enumerate(decoded_pred[0]):
        print('{}위 : {} ({:.2f}%)'.format(i+1, instance[1],instance[2]*100))
   
 
pred_img(images[30])




