import matplotlib.pyplot as plt
import cv2
from img_Resnet_Model import resnet50_pre
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from img_open import img_save


def pred_img(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    img_resized = cv2.resize(img,(224,224))
    pred = resnet50_pre.predict(img_resized.reshape([1,224,224,3]))
    decoded_pred = decode_predictions(pred)

    for i, instance in enumerate(decoded_pred[0]):
        print('{}위 : {} ({:.2f}%)'.format(i+1, instance[1],instance[2]*100))

