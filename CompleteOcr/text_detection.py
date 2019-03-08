from .crnn_model import CRNN
from .east_model import EAST

class OCR:
	def __init__(self):

		self.east = EAST()
		self.crnn = CRNN()

	def predict(self, image):
		
		crop, coords = self.east.predict(image)
		pred = self.crnn.predict(crop)

		return {
			'img': image,
			'cordinates': coords,
			'text': pred
		}
