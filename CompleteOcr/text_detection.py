from .crnn_model import CRNN
from .east_model import EAST

import cv2

class OCR:
	def __init__(self):

		self.east = EAST()
		self.crnn = CRNN()

	def predict(self, path):
		print('First get through EAST model')
		image = cv2.imread(path)
		crop, coords = self.east.predict(image)
		print('Now time for CRNN model')
		pred = self.crnn.predict(crop)
		return {
			'img': image,
			'cordinates': coords,
			'text': pred
		}
