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
    img = cv.resize(img, (224, 224))
    images.append(img)
images = np.stack(images)/255

r = images[:, :, :, 0]
g = images[:, :, :, 1]
b = images[:, :, :, 2]

r = (r - 0.3546618327412568)/(0.22170194987580907)
g = (g - 0.43602398141192555)/(0.23255982872023256)
b = (b - 0.44736975326109596)/(0.23569286717288443)

images = np.stack([r, g, b], axis=3)
np.save("test_set.npy", images)