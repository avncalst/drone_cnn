#========================import libs===========================================================

import threading
import numpy as np
import pathlib
import os
import cv2
import time
import PIL.Image as Image
# import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

from imutils.video import FPS
# from djitellopy import Tello
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file
from pycoral.adapters import detect
from pycoral.adapters import common
# from pycoral.adapters import classify


# from mss import mss
# from pynput.mouse import Listener
from dronekit import connect, VehicleMode
from pymavlink import mavutil

script_dir = str(pathlib.Path(__file__).parent.absolute())
path = '/tflite/'
# print('script_path:',script_dir)

model_file = os.path.join(script_dir+path, 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite')
label_file = os.path.join(script_dir+path, 'coco_labels.txt')
labels = read_label_file(label_file)
# print('labels:',labels)
# print(multiprocessing.cpu_count())

#========================settings==============================================================

confThreshold = 0.65
confidence = 0

#=========================connect drone========================================================
def drone_connect():
	global vehicle
	# vehicle = connect('udpin:192.168.1.30:14550', wait_ready=False)
	vehicle = connect('udpin:192.168.42.14:14550', wait_ready=False)
	vehicle.initialize(8,30)
	vehicle.wait_ready('autopilot_version')

	print("Attitude: %s" % vehicle.attitude)
	print('pitch: %s' %vehicle.attitude.pitch)
	print("Velocity: %s" % vehicle.velocity)
	print("GPS: %s" % vehicle.gps_0)
	print("Flight mode currently: %s" % vehicle.mode.name)
	print('battery level: %s' % vehicle.battery.level)
	print('status: %s'% vehicle.system_status.state)
	print('armed: %s'% vehicle.armed)
	print('heading: %s' %vehicle.heading)
	print('speed: %s'% vehicle.groundspeed)
	print('gps fix: %s'% vehicle.gps_0.fix_type)
	print('sattelites: %s' %vehicle.gps_0.satellites_visible)
	print('location global lat: %s'% vehicle.location.global_frame.lat)
	print('location global lon: %s'% vehicle.location.global_frame.lon)
	print ('location global alt: %s' %vehicle.location.global_frame.alt)
	print('heading: %s'% vehicle.heading)
	print('groundspeed: %s' % vehicle.groundspeed)
	print('velocity: %s'% vehicle.velocity[2])


#===============================move vehicle===================================================
def goto_position_target_local_ned(north, east, down):
	"""
	Send SET_POSITION_TARGET_LOCAL_NED command to request the vehicle fly to a specified
	location in the North, East, Down frame.
	"""
	msg = vehicle.message_factory.set_position_target_local_ned_encode(
		0,       # time_boot_ms (not used)
		0, 0,    # target system, target component
		mavutil.mavlink.MAV_FRAME_LOCAL_NED, # frame
		0b0000111111111000, # type_mask (only positions enabled)
		north, east, down,
		0, 0, 0, # x, y, z velocity in m/s  (not used)
		0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
		0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)
	# send command to vehicle
	vehicle.send_mavlink(msg)  


#================================object detect=================================================
def object_frame():
	
	interpreter = make_interpreter(model_file)
	interpreter.allocate_tensors()
	gst_str = ('udpsrc port=5600 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264"'
	# ' ! rtpjitterbuffer'
	' ! queue'
	' ! parsebin'
	' ! decodebin'
	' ! videoconvert'
	' ! appsink emit-signals=true sync=false max-buffers=1 drop=true' )
	cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
	# cap = cv2.VideoCapture('./video/IA_video.avi')

	# landscape
	width = 640
	height = 360
	fileOut = cv2.VideoWriter('dji.avi',cv2.VideoWriter_fourcc(*'XVID'), 16, (width,height))
	state = 'fly'
	labeltxt = None
	global y 
	# keep looping
	fps = FPS().start()
	while True:
		
		ret,frame = cap.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		imag = cv2.resize(frame, (300,300)) 
		common.set_input(interpreter, imag)
		interpreter.invoke()
		scale = (1,1)
		objects = detect.get_objects(interpreter,confThreshold,scale)
		# print('objects:',objects)

		data_out = []

		if objects:
			for obj in objects:
				inference = [] # clear inference
				box = obj.bbox
				inference.extend((obj.id,obj.score,box))
				# print('inference:',inference)
				data_out.append(inference) # list of all detected objects
			# print('data_out:',data_out)
			objID = data_out[0][0] # object with largest confidence selected
			confidence = data_out[0][1]
			labeltxt = labels[objID]
			box = data_out[0][2]		
			if confidence > confThreshold :
				left,right,top,bottom = draw_rect(imag,box,confidence,labeltxt,use_normalized_coordinates=False)

		else:
			left,right,top,bottom = [0,0,0,0]

		if labeltxt == 'person':
			w = int(right -left)
			h = int(bottom - top)
			# print('h,w:',h,w)

			if state == 'avoid':
				# if left > 100:
				# 	print('go left')
				# else:
				# 	print('go right')
				if h < 135:
					state = 'fly'
					event.clear()
			else:		
				if h > 150:
					state = 'avoid'
					event.set()
					if left > 100:
						y = -0.8
						print('go left')
					else:
						y = 0.8
						print('go right')

		outputDet = cv2.resize(imag, (width,height))
		outputDet = cv2.cvtColor(outputDet, cv2.COLOR_RGB2BGR)
		cv2.putText(outputDet, state, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)		 
		cv2.imshow('frame',  outputDet)
		fileOut.write(outputDet)
		fps.update()
		k = cv2.waitKey(30) & 0xff # press ESC to exit
		if k == 27 :
			break
	fps.stop()
	fileOut.release()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

#================================draw rectangle object=========================================
def draw_rect(image, box,score,class_name,use_normalized_coordinates):
	ymin = box[1]
	xmin = box[0]
	ymax = box[3]
	xmax = box[2]
	# print('draw rectangle')
	color='chartreuse'
	thickness=4
	# use_normalized_coordinates=True
	# class_name = 'avc'
	display_str = '{}:{}%'.format(class_name,round(100*score))
	display_str_list = (display_str,)
	# print('display:',display_str_list) 

	image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
	draw = ImageDraw.Draw(image_pil)
	im_width, im_height = image_pil.size
	if use_normalized_coordinates:
		(left, right, top, bottom) = (xmin * im_width, xmax * im_width,
								  ymin * im_height, ymax * im_height)
	else:
		(left, right, top, bottom) = (xmin, xmax, ymin, ymax)
	if thickness > 0:
		draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
					(left, top)],
					width=thickness,
					fill=color)
	
	font = ImageFont.load_default()

	# If the total height of the display strings added to the top of the bounding
	# box exceeds the top of the image, stack the strings below the bounding box
	# instead of above.
	display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
	# Each display_str has a top and bottom margin of 0.05x.
	total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

	if top > total_display_str_height:
		text_bottom = top
	else:
		text_bottom = bottom + total_display_str_height
	# Reverse list and print from bottom to top.
	for display_str in display_str_list[::-1]:
		text_width, text_height = font.getsize(display_str)
		margin = np.ceil(0.05 * text_height)
		draw.rectangle(
			[(left, text_bottom - text_height - 2 * margin),
			(left + text_width,
			text_bottom)],
			fill=color)
		draw.text(
			(left + margin, text_bottom - text_height - margin),
			display_str,
			fill='black',
			font=font)
		text_bottom -= text_height - 2 * margin

	np.copyto(image, np.array(image_pil))

	return left,right,top,bottom

#=================================object avoid calculation=====================================

def movDrone():

	txt = input("if take-off, hit key to continue")
	print('text:',txt)
	if vehicle.armed==True:

		#-------------------------------------------------
		if txt == 't':
			counter=0
			while counter<5:
				goto_position_target_local_ned(0,0.5,0)
				time.sleep(2)
				print("Moving right")
				counter=counter+1

			time.sleep(2)

			counter=0
			while counter<5:
				goto_position_target_local_ned(0,-0.5,0)
				time.sleep(2)
				print("Moving left")
				counter=counter+1

			time.sleep(2)

			print('landing')
			vehicle.mode=VehicleMode("LAND")
			while not vehicle.mode.name=='LAND':
				print('Waiting for mode LAND')
				time.sleep(0.5)

		#-------------------------------------------------
		else:
			print('thread_movDrone started')
			x = 0 # m/s
			z = 0
			while True:
				event.wait()
				# print('event wait')
				flight_mode = vehicle.mode.name
				print('flight_mode @ object avoidance:',flight_mode)
				count = 0
				while event.is_set():
					print('y:',y)
					goto_position_target_local_ned(x, y, z)
					time.sleep(2)
					count +=1

					if count >= 10:
						print('landing')
						vehicle.mode=VehicleMode("LAND")
						while not vehicle.mode.name=='LAND':
							print('Waiting for mode LAND')
							time.sleep(0.5)
						break

				print('move stopped')
		

#================================main==========================================================

drone_connect()

print("[INFO] starting threads...")
event = threading.Event()
t1 = threading.Thread(target=object_frame)
t2 = threading.Thread(target=movDrone)
t2.daemon=True # has to run in console
t1.start()
t2.start()
t1.join()
	

