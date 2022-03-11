
import numpy as np
from art.estimators.classification import KerasClassifier, TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod
import cv2
from tensorflow.keras.applications import ResNet50V2
import tensorflow as tf
from tensorflow import keras

if False:
    tf.compat.v1.disable_eager_execution()
    model = ResNet50V2(weights="imagenet")
    classifier = KerasClassifier(model=model,clip_values=(0,255))
    attack = FastGradientMethod(estimator=classifier,eps=2)

    image = cv2.imread("siamese-cat-cover.jpg")
    image = cv2.resize(image,(224,224))
    image_save = image.copy()

    image = np.expand_dims(image,0)
    image = image.astype("float32")

    y = np.argmax(classifier.predict(image)[0])
    print(y)
    adver = attack.generate(x=image)
    y_pred = np.argmax(classifier.predict(adver)[0])
    print(y_pred)

    adver_show = np.squeeze(adver,0)
    adver_show = adver_show.astype("uint8")
    cv2.imshow(f"Image Attacked - Predicted : {y_pred}",adver_show)
    cv2.imshow(f"Image Original - Predicted : {y}",image_save)
    cv2.waitKey(0)


######################################################################################
tf.compat.v1.disable_eager_execution()
model = keras.models.load_model("models/cnnClassifierTuned1.h5")

image = cv2.imread("toolbox_runs/1646234255_runs/10.jpg",0)
image_save = image.copy()
image = np.expand_dims(image,0)
image = np.expand_dims(image,-1)
image = image.astype("float32")


classifier = KerasClassifier(model=model,clip_values=(0,255), use_logits=False)
attack = FastGradientMethod(estimator=classifier,eps=2)
y = np.argmax(classifier.predict(image))
print(y)
adver = attack.generate(x=image)
y_pred = np.argmax(classifier.predict(adver))
print(y_pred)



adver_show = np.squeeze(adver,-1)
adver_show = np.squeeze(adver_show,0)
adver_show = adver_show.astype("uint8")
adver_show = cv2.resize(adver_show, (100, 100))
image_save = cv2.resize(image_save, (100, 100))
cv2.imshow(f"Image Attacked - Predicted : {y_pred}",adver_show)
cv2.imshow(f"Image Original - Predicted : {y}",image_save)
cv2.waitKey(0)
