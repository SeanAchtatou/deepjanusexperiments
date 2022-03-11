import cv2
import os

runs_directory = "runs"

runs = os.listdir(runs_directory)

for i in runs:
    files = os.listdir(runs_directory+"/"+i)
    for j in files:
        if j == "archive":
            image = os.listdir(runs_directory+"/"+i+"/"+j)
            for k in image:
                if "png" in k:
                    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
                    img = cv2.imread(runs_directory+"/"+i+"/"+j+"/"+k,0)
                    imS = cv2.resize(img, (1280, 720))
                    cv2.imshow('Image',img)
                    cv2.waitKey(0)

