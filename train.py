import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
from unet import UNet

torch.cuda.set_device(0)
device = torch.device('cuda')
loss_function = nn.CrossEntropyLoss()

images = torch.tensor(np.load("images.npy"), dtype=torch.float32)
masks = torch.tensor(np.load("masks.npy"), dtype=torch.float32)

split = int(images.shape[0]*0.9)
train_images = images[:split]
train_labels = masks[:split]

val_images = images[split:]
val_labels = masks[split:]

print(train_labels.shape)
print(val_labels.shape)

model = UNet(num_classes=1, in_channels=3, depth=5, merge_mode='concat')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.to(device)

n_epochs = 10
batch_size = 8

for epoch in range(n_epochs):
	permutation = torch.randperm(train_images.shape[0])
	for i in range(0, permutation.shape[0], batch_size):
		optimizer.zero_grad()

		indices = permutation[i:i+batch_size]
		x, y = train_images[indices], train_labels[indices]

		x = x.to(device)
		y = y.to(device)

		logits = model(x.permute(0, 3, 1, 2))
		loss = loss_function(logits, y)
		print(loss.item())
		
		loss.backward()
		optimizer.step()