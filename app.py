from flask import Flask
from flask import request
from .CompleteOcr.text_detection import OCR
import timeit
app = Flask(__name__)

print('Let\'s load the model first, hope our system does not crash.')
start = timeit.timeit()
OCR.init()
end = timeit.timeit()
print('Model loaded in - ' + str(end - start) + ',cool!')


@app.route('/', methods=['POST'])
def index(self):
	img = request.form['img']
	print('We got request from frontend to process image, lets predict on it :)')

from PIL import image
import re
from io import BytesIO
import base64
import numpy as np
def decode(data):
	image_data = re.sub('^data:image/.+;base64,', '', data)
	im = Image.open(BytesIO(base64.b64decode(image_data)))
	pix = np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
	
