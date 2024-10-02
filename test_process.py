import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

images = []
directory = "test\\"
for file in os.listdir(directory):
    img = cv.imread(os.path.join(directory, file))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (224, 224))
    images.append(img)
images = np.stack(images)/255

r = images[:, :, :, 0]
g = images[:, :, :, 1]
b = images[:, :, :, 2]

r = (r - 0.485)/(0.229)
g = (g - 0.456)/(0.224)
b = (b - 0.406)/(0.225)

images = np.stack([r, g, b], axis=3)
np.save("test_set.npy", images)