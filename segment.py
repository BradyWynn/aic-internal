import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.ops as ops
import cv2 as cv

masks = torch.tensor(np.load("masks.npy"))
images = torch.tensor(np.load("images.npy"))

cropped_images = []
for i in range(images.shape[0]):
    points = ops.masks_to_boxes(masks[i].unsqueeze(0)).int().tolist()[0]
    img = images[i][points[1]:points[3], points[0]:points[2]]
    img = cv.resize(img.numpy(), (64, 64))
    cropped_images.append(img)
cropped_images = np.stack(cropped_images)
np.save("cropped_images.npy", cropped_images)