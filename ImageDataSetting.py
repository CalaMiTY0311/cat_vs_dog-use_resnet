from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from dogsVcats_copy import copy_train_path
from keras_model import image_height, batch_size, input_shape, epochs, cnn_api
from make_callback import callbacks
import os

import tensorflow as tf


from Add_GPU import GPU_SETTING
#CPU -> GPU 셋팅
GPU_SETTING()


train_datagen = ImageDataGenerator(rescale = 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255) # 검증데이터 스케일 조정만 합니다.

train_generator = train_datagen.flow_from_directory(
    os.path.join(copy_train_path,"train"),
    target_size = (image_height, image_height),
    batch_size = batch_size,
    class_mode = "binary"
    )
validation_generator = val_datagen.flow_from_directory(
    os.path.join(copy_train_path,"validation"),
    target_size = (image_height, image_height),
    batch_size = batch_size,
    class_mode = "binary"
    )
    
newType_model = cnn_api(input_shape)

with tf.device("/gpu:0"):
    hist = newType_model.fit(train_generator, steps_per_epoch = 20000//batch_size, epochs= epochs,
                            validation_data = validation_generator, validation_steps = 5000//batch_size,
                            callbacks = callbacks)

import matplotlib.pyplot as plt
train_acc = hist.history['acc']
val_acc = hist.history['val_acc']

train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(1,len(train_acc)+1)

plt.plot(epochs,train_acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'r',label='Val acc')
plt.title('Training and Val accuracy')
plt.legend()

plt.figure()
plt.plot(epochs,train_loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'r',label='Val loss')
plt.title('Training and Val loss')
plt.legend()

plt.show()