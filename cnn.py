import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.layers = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(5, 5)),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(3, 3)),
			nn.ReLU(),
			nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(3, 3)),
			nn.ReLU(),
			nn.Flatten(start_dim=1, end_dim=-1),
			nn.Linear(in_features=256, out_features=4, bias=True)
			)
		
		self.reset_params()
		
	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.xavier_uniform_(m.weight)
			nn.init.constant_(m.bias, 0)

	def reset_params(self):
		for i, m in enumerate(self.modules()):
			self.weight_init(m)
			
	def forward(self, x):
		return self.layers(x)