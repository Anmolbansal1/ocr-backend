import os
import cv2
import string
from tqdm import tqdm
import numpy as np

import torch
from torch.autograd import Variable

from .dataset.collate_fn import text_collate
from .dataset.data_transform import Resize, Rotation, Translation, Scale
from .model_loader import load_model
from torchvision.transforms import Compose

import string

import editdistance

label = ''
backend = 'resnet18'
snapshot = './CompleteOcr/snaps/crnn_best'
input_size = [320, 32]
seq_proj = [10, 20]
abc = string.digits+string.ascii_letters

transform = Compose([
	Resize(size=(input_size[0], input_size[1]))
])

class CRNN:
	def __init__(self):
		print('Now time for crnn')
		self.net = load_model(abc, seq_proj, backend, snapshot, cuda=True).eval()
		self.transform = Compose([
			# Rotation(),
			Resize(size=(input_size[0], input_size[1]))
		])
		print('Loadedededed')
	
	def predict(self, imgs):
		print('time for crnn')
		text = []
		for img in imgs:
			img = self.transform(img)
			img = torch.from_numpy(img.transpose((2, 0, 1))).float().unsqueeze(0)
			img = Variable(img)
			out = self.net(img, decode=True)
			text.append(out)
		
		return text




