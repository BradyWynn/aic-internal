import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

images = []
labels = []
directory = "train\\"
for folder in os.listdir(directory):
    inner_directory = os.path.join(directory, folder)
    for file in os.listdir(inner_directory):
        img = cv.imread(os.path.join(inner_directory, file))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (224, 224))
        images.append(img)
        images.append(cv.rotate(img, cv.ROTATE_90_CLOCKWISE))
        images.append(cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE))
        images.append(cv.rotate(img, cv.ROTATE_180))
        labels.append(int(folder))
        labels.append(int(folder))
        labels.append(int(folder))
        labels.append(int(folder))
images = np.stack(images)/255
labels = np.stack(labels)

r = images[:, :, :, 0]
g = images[:, :, :, 1]
b = images[:, :, :, 2]

r = (r - 0.485)/(0.229)
g = (g - 0.456)/(0.224)
b = (b - 0.406)/(0.225)

images = np.stack([r, g, b], axis=3)
np.save("images.npy", images)
np.save("labels.npy", labels)