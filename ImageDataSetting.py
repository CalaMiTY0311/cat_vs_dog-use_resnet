from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from dogsVcats_copy import copy_train_path
from keras_model import image_height, batch_size, input_shape, epochs, cnn_api
from make_callback import callbacks
import os

import tensorflow as tf
#CPU -> GPU 셋팅
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

        


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

#gpu가 두개로 병렬로 각각 다른 작업을 원하시면
# with를 사용하시고 gpu:0 , gpu:1 이러게 모델을 지정하시면 됩니다. 
# from tensorflow as tf
# with tf.device('gpu:0'):
#     newType_model = cnn_api(input_shape)
    
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