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
        img = cv.resize(img, (224, 224))
        images.append(img)
        labels.append(int(folder))
images = np.stack(images)/255
labels = np.stack(labels)

r = images[:, :, :, 0]
g = images[:, :, :, 1]
b = images[:, :, :, 2]

print(r.mean(), r.std())
print(g.mean(), g.std())
print(b.mean(), b.std())

r = (r - r.mean())/(r.std())
g = (g - g.mean())/(g.std())
b = (b - b.mean())/(b.std())

images = np.stack([r, g, b], axis=3)
np.save("images.npy", images)
np.save("labels.npy", labels)