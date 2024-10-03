import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 15))

index = 508

images = np.load("images.npy")
ax1.imshow(images[index])

masks = np.load("masks.npy")
ax2.imshow(masks[index])

plt.show()

# masks = []
# directory = "masks\\"
# for folder in os.listdir(directory):
#     inner_directory = os.path.join(directory, folder)
#     for file in os.listdir(inner_directory):
#         img = np.load(os.path.join(inner_directory, file))
#         masks.append(np.rot90(img, 0))
#         masks.append(np.rot90(img, 1, axes=(1, 0)))
#         masks.append(np.rot90(img, 1, axes=(0, 1)))
#         masks.append(np.rot90(img, 2))
# masks = np.stack(masks)

# print(masks.shape)

# np.save("masks.npy", masks)