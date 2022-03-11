import random
import numpy as np
import shutil
import art.attacks.evasion as attacks
import os
import h5py
import art
import tensorflow as tf
import cv2
import time

from matplotlib import pyplot as plt
from art.attacks.evasion import ProjectedGradientDescent, FastGradientMethod
from art.estimators.classification import KerasClassifier as KC
from tensorflow import keras
from art.utils import load_mnist
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from config import num_classes, DATASET

tf.compat.v1.disable_eager_execution()
print("Retrieving data...")
(x_train, y_train), (x_test_C, y_test_C), min_pixel_value, max_pixel_value = load_mnist()
rnd = np.random.choice(x_test_C.shape[0],500)
x_test_M = x_test_C[rnd]
y_test_M = y_test_C[rnd]

print("Retrieving models...")
models = []
classifiers = []
for i in os.listdir("models"):
    if (("FineTuned" in i) and (".h5" in i)):
        model = keras.models.load_model("models/"+i)
        models.append(model)
        classifier = KC(model=model, clip_values=(0,255))
        classifiers.append(classifier)
       
for i in range(len(models)):
    print(f"Model {i}: {models[i]}")
    print(f"Classifier {i}: {classifiers[i]}")

if "toolbox_runs" not in os.listdir():
    os.mkdir("toolbox_runs")

for i in os.listdir("toolbox_runs"):
    shutil.rmtree(os.path.join("toolbox_runs",i))

print("Generating attacks...")
path_runs = ""
for i in classifiers:
    path_runs = f"{str(time.time()*1000)[:10]}_runs"
    if path_runs not in os.listdir("toolbox_runs"):
        os.mkdir(os.path.join("toolbox_runs",path_runs))
    attack = ProjectedGradientDescent(estimator=i, eps=2)
    #attack = FastGradientMethod(estimator=i, eps=2)
    adversarial_examples = attack.generate(x=x_test_M)
    final_examples = np.squeeze(adversarial_examples,-1)
    count = 0
    imgs = []
    for j in final_examples:
        imgs.append(j)
        cv2.imwrite(os.path.join("toolbox_runs"+"/"+path_runs,f"{count}.jpg"),j)
        count += 1

    imgs = np.asarray(imgs)
    imgs = imgs.reshape(imgs.shape[0], 28, 28, 1)
    imgs = imgs.astype('float32')
    print(imgs.shape)

    print(f"Labels:       {np.argmax(y_test_M,axis=1)[:20]}")
    test = i.predict(x_test_M)
    print(f"Predicted:    {np.argmax(test,axis=1)[:20]}")
    accuracyB = np.sum(np.argmax(test, axis=1) == np.argmax(y_test_M, axis=1)) / len(y_test_M)
    results = i.predict(adversarial_examples)
    print(f"Attack:       {np.argmax(results,axis=1)[:20]}")
    accuracyA = np.sum(np.argmax(results, axis=1) == np.argmax(y_test_M, axis=1)) / len(y_test_M)
    print("Accuracy on benign: {}%".format(accuracyB * 100))
    print("Accuracy on attack: {}%".format(accuracyA * 100))
    input("Waiting for next...")
    '''
    for j in results:
        cv2.namedWindow(f"Predicted - {np.argmax(j)}", cv2.WINDOW_NORMAL)
        img = cv2.imread("toolbox_runs"+"/"+ path_runs + "/" + f"{count}.jpg",0)
        imS = cv2.resize(img, (1280, 720))
        cv2.imshow(f"Predicted - {np.argmax(j)}",imS)
        cv2.waitKey(0)
        if cv2.waitKey(0) == ord("q"):
            cv2.destroyAllWindows()
            break
        cv2.destroyAllWindows()
        count += 1
    #input("Waiting for next models evaluation...")
    countModel += 1'''




