from multiprocessing import Process
from multiprocessing import Queue

import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import time
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

from imutils.video import FPS
from djitellopy import Tello
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import detect
from pycoral.adapters import common
from pycoral.adapters import classify
from simple_pid import PID

tello_drone = True
cust = False

tpu = True
counter = 1 # 1: no flight; 0: flight Tello
confThreshold = 0.25
test = False

# print(multiprocessing.cpu_count())


TFLITE_PATH = 'Tensorflow/tflite/'
img = None
out = None 
k = 0
flag = 0
cx = 0
cy = 0
area = 0
fps = 0.0
qfps = 0.0

#----------------------------------------------------------------------------------------------
def object_frame(inputQueue,outputQueue):
	# interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH+'/model.tflite')
	if not tpu:
		interpreter = tflite.Interpreter(model_path=TFLITE_PATH+'/model.tflite')
	else:
		if not cust:
			interpreter = make_interpreter(TFLITE_PATH+\
				'/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite')
		if cust:
			interpreter = make_interpreter(TFLITE_PATH+\
				'/detect_edgetpu.tflite')
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	 
	# keep looping
	while True:
		data_out = [] 
		# check to see if there is a frame in our input queue
		if not inputQueue.empty():
			# grab the frame from the input queue
			img = inputQueue.get()

			if not tpu: 
				input_data = np.expand_dims(img,axis=0)
				input_data = input_data/127.5-1
				input_data = np.asarray(input_data, dtype=np.float32)
				interpreter.set_tensor(input_details[0]['index'], input_data)
				interpreter.invoke()
			else:
				common.set_input(interpreter, img)
				interpreter.invoke()
				scale = (1,1)
				objects = detect.get_objects(interpreter,confThreshold,scale)

			if not tpu:
				boxes = interpreter.get_tensor(output_details[0]['index'])[0]
				classe = interpreter.get_tensor(output_details[1]['index'])[0]
				score = interpreter.get_tensor(output_details[2]['index'])[0]
				data_out = [boxes,classe,score]
			else:
				if objects:
					for obj in objects:
						box = obj.bbox
						# print('bbox:',obj.bbox)
						xmin = int(box[0])
						ymin = int(box[1])
						xmax = int(box[2])
						ymax = int(box[3])
						data_out = [[[ymin,xmin,ymax,xmax]],obj.id,obj.score]
			
			# print('data_out:',data_out )
				
			outputQueue.put(data_out)

#----------------------------------------------------------------------------------------------
def sav_frame(vidQueue):
	 
	# keep looping
	while True:
		if not vidQueue.empty():
			image = vidQueue.get()
			fileOut.write(image)
#----------------------------------------------------------------------------------------------
def draw_rect(image, box,score,class_name,use_normalized_coordinates):
	ymin = box[0]
	xmin = box[1]
	ymax = box[2]
	xmax = box[3]
	# print('draw rectangle')
	color='chartreuse'
	thickness=4
	# use_normalized_coordinates=True
	# class_name = 'avc'
	display_str = '{}:{}%'.format(class_name,round(100*score))
	display_str_list = (display_str,) 

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
			[(left, text_bottom - text_height - 2 * margin), (left + text_width,
																text_bottom)],
			fill=color)
		draw.text(
			(left + margin, text_bottom - text_height - margin),
			display_str,
			fill='black',
			font=font)
		text_bottom -= text_height - 2 * margin

	np.copyto(image, np.array(image_pil))

	w = int(right -left)
	h = int(bottom - top)
	
	cx = int(left) + w//2
	cy = int(top) + h//2
	area = w*h
	return cx,cy,area

#----------------------------------------------------------------------------------------------

def traceDrone(cx,cy,area):
	width = 320 # image size used @ inference
	height = 320
	error = width//2 - cx
	errora = (7000-area)//100 # scale error
	errorb = height//2 - cy 
	 
	pid_0 = PID(Kp=0.5,Kd=0.5,setpoint=width//2,output_limits=(-50,50))			# yaw
	pid_1 = PID(Kp=0.5,Kd=0.5,setpoint=7000//100,output_limits=(-30,30))		# pitch
	pid_2 = PID(Kp=0.5,Kd=0.5,setpoint=height//2,output_limits=(-30,30))		# throttle
	speed_0 = -int(pid_0(cx))
	speed_1 = int(pid_1(area//100))
	speed_2 = int(pid_2(cy))

	print('speed:',speed_0)
	print('speeda:',speed_1)
	print('speedb:',speed_2)
	
	print('error:',error)
	print('errora:',errora)
	print('errorb:',errorb)
	
	if cx:
		drone.yaw_velocity = speed_0
		drone.for_back_velocity = speed_1
		drone.up_down_velocity = speed_2
	else:
		drone.for_back_velocity = 0
		drone.left_right_velocity = 0
		drone.up_down_velocity = 0
		drone.yaw_velocity = 0
		drone.speed = 0
		error = 0
		errora = 0
	if drone.send_rc_control and not test:
		drone.send_rc_control(drone.left_right_velocity,drone.for_back_velocity,
							drone.up_down_velocity,drone.yaw_velocity)
	

#----------------------------------------------------------------------------------------------
def test_tello():
	drone.move_left(20)
	time.sleep(3)
	drone.rotate_clockwise(90)
	time.sleep(3)
	drone.move_forward(10)
#----------------------------------------------------------------------------------------------

if tello_drone:
	# init Tello
	drone = Tello()
	drone.connect()
	drone.for_back_velocity = 0
	drone.left_right_velocity = 0
	drone.up_down_velocity = 0
	drone.yaw_velocity = 0
	drone.speed = 0
	print(drone.get_battery())
	drone.streamoff()
	drone.streamon()
	
		
else:
	cap = cv2.VideoCapture(0)

width = 640
height = 480

fileOut = cv2.VideoWriter('detectionTello.avi',cv2.VideoWriter_fourcc(*'XVID'), 20, (width,height))
label_id_offset = 1
cap_class_id = 1
category_index = {cap_class_id: {'id': cap_class_id, 'name': 'avc'}}
dic = category_index[cap_class_id]
class_name = dic['name']

inputQueue = Queue()
outputQueue = Queue()
vidQueue = Queue()

print("[INFO] starting process...")
p = Process(target=object_frame, args=(inputQueue,outputQueue,))
pvid = Process(target=sav_frame, args=(vidQueue,))
p.daemon = True
pvid.daemon = True
p.start()
pvid.start()

#time the frame rate....
timer1 = time.time()
frames = 0
queuepulls = 0
timer2 = 0
t2secs = 0


fps1 = FPS().start()

while True:
	if tello_drone:
		frame_read = drone.get_frame_read()
		myFrame = frame_read.frame
		if counter == 0:
			drone.takeoff()
			time.sleep(8)
			print('starting tello')
			drone.move_up(80)
			time.sleep(3)
			
			if test:
				print('testing tello')
				test_tello()
			counter = 1
		
	else:
		ret, myFrame = cap.read()
	
	img = cv2.cvtColor(myFrame, cv2.COLOR_BGR2RGB)
	if not cust:
		new_img = cv2.resize(img,(320, 320))
	if cust:
		new_img = cv2.resize(img,(300, 300))

	if queuepulls ==1:
		timer2 = time.time()

	# print('Queueinput:',inputQueue.empty())	
	if inputQueue.empty():
		inputQueue.put(new_img)

	# print('Queueoutput:',outputQueue.empty())
	if not outputQueue.empty():
		out = outputQueue.get()
		# print('out:',out)
		
		
		try:
			
			box = out[0]
			clas = out[1]
			scor = out[2]

			if not tpu:	
				clas = clas.astype(np.int64) #classe should be integers
				for index,score in enumerate(scor):
					# select only box with score > threshold
					if score > confThreshold:
						cx,cy,area = draw_rect(new_img,box[0],score,class_name,use_normalized_coordinates=True)
			else:
				if scor > confThreshold:
					cx,cy,area = draw_rect(new_img,box[0],scor,class_name,use_normalized_coordinates=False)
					
		except:
			pass
		
		# print('area:',area) 

		if tello_drone:	
			# print('cx,cy,area:',cx,cy,area)
			traceDrone(cx,cy,area)
			cx = 0
			cy = 0
			area = 0
		
		outputDet = cv2.resize(new_img, (width,height))
		outputDet = cv2.cvtColor(outputDet, cv2.COLOR_RGB2BGR)		 
		cv2.imshow('frame',  outputDet)
		# fileOut.write(outputDet)
		if inputQueue.empty():
			vidQueue.put(outputDet)
		k = cv2.waitKey(30) & 0xff # press ESC to exit
		fps1.update()
		queuepulls += 1
	if k == 27 :
		if tello_drone:
			drone.land()
		break
		
	
	# FPS calculation
	frames += 1
	if frames >= 1:
		end1 = time.time()
		t1secs = end1-timer1
		fps = round(frames/t1secs,2)
	
	if queuepulls > 1:
		end2 = time.time()
		t2secs = end2-timer2
		qfps = round(queuepulls/t2secs,2)

	print('vid-fps:',fps)
	print('tpu-qfps:',qfps)
fps1.stop()
print("[INFO] elapsed time: {:.2f}".format(fps1.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps1.fps()))
cv2.destroyAllWindows()

if not tello_drone:
	cap.release()
	time.sleep(2)
fileOut.release()

if __name__ == '__main__':
	if tello_drone:
		drone.streamoff()
		time.sleep(3)
	p.terminate()
	p.join()
	pvid.terminate()
	pvid.join()
	inputQueue.close()
	outputQueue.close()
	vidQueue.close()
	print('end program')

	

