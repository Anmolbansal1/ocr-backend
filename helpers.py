from PIL import Image
import re
from io import BytesIO
import base64
import numpy as np

def decode(num, data):
	image_data = re.sub('^data:image/.+;base64,', '', data)
	im = Image.open(BytesIO(base64.b64decode(image_data)))
	path = './temp/img_' + str(num) + '.png'
	im.save(path)
	return path
	# pix = np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
	# return pix
	