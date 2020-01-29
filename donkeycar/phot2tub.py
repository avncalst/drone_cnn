from donkeycar.vehicle import Vehicle
import numpy as np
from donkeycar.parts.datastore import TubHandler
from tensorflow.keras.preprocessing.image import img_to_array
from imutils import paths
import cv2
import os
import time

V=Vehicle()

class foto(object):
	def __init__(self):

		print('start foto...')
		self.angle = 0.0
		self.throttle = 0.0
		self.image_array = None
		self.mode = 'user'
		self.data = []
		self.labels = []
		self.idx = 0
		self.recording = False
		self.vehicle = None
		self.running = True
		self.imagePaths = sorted(list(paths.list_images('./mycar/phot')))
		# print(self.imagePaths)
		# loop over the input images
		for imagePath in self.imagePaths:
			# load the image, pre-process it, and store it in the data list
			image = cv2.imread(imagePath)
			image = cv2.resize(image, (160, 120))
			image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
			# cv2.imshow('foto', image)
			# cv2.waitKey(1)
			# image = img_to_array(image)
			self.data.append(image)

			# extract the class label from the image path and update the
			# labels list
			label = imagePath.split(os.path.sep)[-2]
			if label == "stop":
				label = 0.9
			if label == "left":
				label = -0.3 
			if label == "right":
				label = 0.3
			if label == "fly":
				label = -0.9
			self.labels.append(label)
		# print(data)
		print('#labels:',len(self.labels))

	def run_threaded(self):
		return self.image_array,self.angle, self.throttle, self.mode, self.recording

	def update(self):
		while self.running:
		
			self.image_array = self.data[self.idx]
			lab = self.labels[self.idx]
  
			# angle values will result in np.array with dim 4 @ training
			if lab == -0.9:
				self.recording = True
				self.angle = -0.9 # fly
				# print('\nfly')
			elif lab == -0.3:
				self.recording = True
				self.angle = -0.3
				# print('\nleft')
			elif lab == 0.3:
				self.recording = True 
				self.angle = 0.3
				# print('\nright')
			elif lab == 0.9:        
				self.recording = True
				self.angle = 0.9
				# print('\nstop')

			time.sleep(0.1) # needed to avoid false records (angle=throttle=0)                
		
			self.recording = False
			self.angle = 0
			self.throttle = 0

			if self.idx < len(self.labels)-1:
				self.idx +=1
			else: 
				self.shutdown()
			# print(self.idx)

	def shutdown(self):
		self.running = False
		time.sleep(1)
	

ctr = foto()
V.add(ctr, 
	inputs=[],
	outputs=['cam/image_array','user/angle', 'user/throttle', 'user/mode', 'recording'],
	threaded=True)


class CvImageView(object):

	def run(self, image):
		if image is None:
			return
		try:
			if(image is not None):
				image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)  #if PiCamera (PICAM) used, not in case of PiCam (PICAM_1)
				cv2.imshow('frame', image)
				# print (image)
				# print('image present')
				cv2.waitKey(1)
		except:
			pass
   

	def shutdown(self):
		cv2.destroyAllWindows() 

disp = CvImageView()
V.add(disp, inputs=["cam/image_array"])

inputs=['cam/image_array',
            'user/angle', 'user/throttle', 
            'user/mode']

types=['image_array',
           'float', 'float',
           'str']

meta=[]

th = TubHandler(path='./mycar/data')
tub = th.new_tub_writer(inputs=inputs, types=types, user_meta=meta)
V.add(tub, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')

V.start()
 