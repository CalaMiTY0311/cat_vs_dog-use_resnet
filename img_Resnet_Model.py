import tensorflow as tf

resnet50_pre = tf.keras.applications.resnet.ResNet50(weights='imagenet', input_shape=(224,224,3))
#resnet50_pre.summary()


#pred_img(images[30])




