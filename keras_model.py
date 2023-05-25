import tensorflow as tf
from keras import Input
from keras import layers ,models, losses ,optimizers

batch_size = 32                                        #데이터를 나눠서 훈련할떄    
no_classes = 1                                          #출력수를 나타내는 변수
epochs = 50                                             #데이터 학습횟수
image_height, image_width = 150,150                     #이미지 사이즈 크기
input_shape = (image_height,image_width,3)              #이미지 사이즈 및 채널 수 채널수 1 = 흑백 채널수 3 = 컬러

def cnn_api(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=1024, activation='relu'),
        tf.keras.layers.Dense(units=no_classes, activation='sigmoid')
    ])
    
    model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), metrics=['acc'])
    return model
