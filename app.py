#!/usr/bin/env python3
# -*- coding: cp1252 -*-


from flask import Flask
from flask import request, Response
from flask_cors import CORS, cross_origin
from .CompleteOcr.text_detection import OCR
import requests
import cv2
import json
from PIL import Image
import numpy as np
from .helpers import decode
import timeit
import base64
app = Flask(__name__)
cors = CORS(app)

print('Let\'s load the model first, hope our system does not crash.')
start = timeit.timeit()
ocr = OCR()
end = timeit.timeit()
print('Model loaded in - ' + str(end - start) + ',cool!')

user_requests = 0
port = 'http://192.168.0.135:5000'

@app.route('/', methods=['POST'])
@cross_origin()
def index():
	try:
		data = request.get_json()
		path = decode(user_requests, data['img'])
		print('We got request from frontend to process image, lets predict on it :)')
		pred = ocr.predict(path)
		print(pred['text'])
 		#  {
		# 	'img': image,
		# 	'cordinates': coords,
		# 	'text': pred
		# }
		print(type(pred['img']))
		result, img_encoded = cv2.imencode('.png', pred['img'])
		# path = './temp/img_' + str(num) + '.png'
		# pred['img'] = 
		
		string = base64.b64encode(img_encoded)
		pred['img'] = string
		# print(string)
		# print('Is image encoded properly? ' + str(result))
		# print(img_encoded)
		r = requests.post(port, params={'coordinates': pred['cordinates'], 'text': pred['text']}, json=pred)
		return Response({'message': 'You are all good', 'text': pred['text']}, mimetype='application/json')

	except Exception as e:
		print(e)
		response = {'message': 'Something bad happened', 'error': e}
		return Response(response, mimetype='application/json')

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=3001, debug=True)
