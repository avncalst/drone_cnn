import threading
from pymavlink import mavutil 
import imutils
import numpy as np
import time
import cv2
import pyrealsense2 as rs
import sys
from operator import itemgetter
from imutils.video import FPS
from dronekit import connect, VehicleMode
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file
from pycoral.adapters import detect
from pycoral.adapters import common


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ---------------------functions------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def slide_velocity(velocity_x, velocity_y, velocity_z):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
    0,       # time_boot_ms (not used)
    0, 0,    # target system, target component
    mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, # frame
    type_mask=mask,
    x=0, y=0, z=0, # x, y, z positions in m (not used)
    vx=velocity_x, vy=velocity_y, vz=velocity_z, # x, y, z velocity in m/s
    afx=0, afy=0, afz=0, # x, y, z acceleration in m/s^2
    yaw=0, yaw_rate=0)   # yaw, yaw_rate in rad, rad/s

    return msg
 

def progress(string):
    print(string, file=sys.stdout)
    sys.stdout.flush()

def distances_depth_image(depth_mat,min_depth_m,max_depth_m,depth_scale,corn_1,corn_2):
    # Parameters for depth image
    step = 2
    lower_pixel = corn_1[1]    # v1
    upper_pixel = corn_2[1]    # v2
    left_pixel = corn_1[0]
    right_pixel = corn_2[0]
    distances_length = (right_pixel-left_pixel)//step
    # print('distances_length:',distances_length)
    distances = np.ones(distances_length, dtype=np.uint16)
    distances = distances*800 # avoid false positives due to dist_m > min_depth_m and dist_m < max_depth_m
    
    # clearance rectangle uv coordinates: corn_1(72,44), corn_2(168,92) corresponds to of about 1.2m x 0.8m @ distance 2m
    

    for i in range(distances_length):
        lower_pixel = corn_1[1]    # v1
        upper_pixel = corn_2[1]    # v2
        # calculate min @ column i*step+left_pixel in slice lower_pixel:upper_pixel
        min_point_in_scan = np.min(depth_mat[lower_pixel:upper_pixel,i * step+left_pixel])
        dist_m = min_point_in_scan * depth_scale

        # Note that dist_m is in meter, while distances[] is in cm.
        if dist_m > min_depth_m and dist_m < max_depth_m:
            distances[i] = dist_m * 100
    return distances 
    
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


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# -----------------------thread-1-----------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def ai():
    # Configure depth and color streams
    test_img = cv2.imread('testImag.jpg')
    pipeline = rs.pipeline()
    config = rs.config()
    colorizer = rs.colorizer()
    config.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 15)
    model_file = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    labels = read_label_file('coco_labels.txt')
    interpreter = make_interpreter(model_file)
    interpreter.allocate_tensors()
##    align = rs.align(rs.stream.color)
    dist_thres_cm = 180 # cm
    max_depth_m = 8
    min_depth_m = 0.1

    confThreshold = 0.65
    confidence = 0   

    off_ul = 15
    off_vl = 10
    off_ur = 25
    off_vr = 15
    
    corn_1 = (72, 44)
    corn_2 = (168, 92)
    cornColor_1 = (corn_1[0]-off_ul, corn_1[1]-off_vl) # compensate # FOV cameras
    cornColor_2 = (corn_2[0]+off_ur, corn_2[1]+off_vr)
    

    filters = [
        [True,  "Decimation Filter",   rs.decimation_filter()],
        [True,  "Threshold Filter",    rs.threshold_filter()],
        [True,  "Depth to Disparity",  rs.disparity_transform(True)],
        [True,  "Spatial Filter",      rs.spatial_filter()],
        [True,  "Temporal Filter",     rs.temporal_filter()],
        [True,  "Hole Filling Filter", rs.hole_filling_filter(True)],
        [True,  "Disparity to Depth",  rs.disparity_transform(False)]
    ]

    if filters[1][0] is True:
        filters[1][2].set_option(rs.option.min_distance, min_depth_m)
        filters[1][2].set_option(rs.option.max_distance, max_depth_m)

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    progress("INFO: Depth scale is: %.4f" % depth_scale)

    progress("INFO: video recording")
    out = cv2.VideoWriter('avcnet.avi',cv2.VideoWriter_fourcc(*'XVID'), 15, (240,136))
    
  
    # default parameters
    orient=0
    tim_old=0.1
    state='fly'
    global velocity_y
    velocity_y=0
    fly_1=400
    k=0.66  # k=(tau/T)/(1+tau/T) tau time constant LPF, T period
            # k=0 # no filtering
    fps = FPS().start()
    # Start streaming
    
    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
##            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Apply the filters
            filtered_frame = depth_frame
            for i in range(len(filters)):
                if filters[i][0] is True:
                    filtered_frame = filters[i][2].process(filtered_frame)

            # Extract depth in matrix form
            depth_data = filtered_frame.as_frame().get_data()
            depth_mat = np.asanyarray(depth_data)   # shape 136,240

            # Convert images to numpy arrays
            # output_image = np.asanyarray(colorizer.colorize(filtered_frame).get_data()) #shape: 136,240,3
            color_image = np.asanyarray(color_frame.get_data())

            # calculate distance
            distances = distances_depth_image(depth_mat,min_depth_m,max_depth_m,depth_scale,corn_1,corn_2)

            # Stack both images horizontally
            # output_color = cv2.resize(color_image, (240, 136))
            if color_frame:
                imag = cv2.resize(color_image, (300,300))
                common.set_input(interpreter, imag)
                interpreter.invoke()
                scale = (1,1)
                objects = detect.get_objects(interpreter,confThreshold,scale)
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
                        draw_rect(imag,box,confidence,labeltxt,use_normalized_coordinates=False)

            output_color = cv2.resize(imag, (240, 136))            
            # cv2.rectangle(output_image, corn_1, corn_2, (0, 255, 0), thickness=2)
            cv2.rectangle(output_color, cornColor_1, cornColor_2, (0, 255, 0), thickness=2) 

#===========================================================================
            # classify the input image
            fly = np.min(distances) # distance in cm
            left = (distances[0:24] < dist_thres_cm).sum()      # object is on the left side
            right = (distances[24:48] < dist_thres_cm).sum()    # object is on the right side
            stop =  ((distances[0:48] < dist_thres_cm).sum() == 48)*48           
            # build the label
            my_dict = {'left':left,'right':right,'stop':stop}       
            maxPair = max(my_dict.items(), key=itemgetter(1))
            fly_f = k*fly_1 + (1-k)*fly
            fly_1 = fly_f
            proba=int(fly_f)

            if state == 'avoid':
                if fly_f >= dist_thres_cm+10:
                    state='fly'
                    print(state,int(fly_f),file=f)
                
            else:
                label='forward'
                # proba=fly_f
                if fly_f <= dist_thres_cm:
                    label=maxPair[0]
                    # proba=int(fly_f)
                    print(my_dict,int(fly_f),file=f)
                    state='avoid'

            label_1 = "{} {} {}".format(label, proba, state)
            # draw the label on the image
            cv2.putText(output_color, label_1, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
            # Write the frame into the file 'output.avi'
            out.write(output_color)

            if vehicle.channels['8'] > 1400:
                if state == "fly":
                    event.clear()
    
                if state == "avoid":
                    event.set()

                    if label == 'left':
                        velocity_y = 0.8
                    if label == 'right':
                        velocity_y = -0.8
                    if label == 'stop':
                        velocity_y = 0
                
            if vehicle.channels['8'] < 1400:
                event.clear()
            

        
            # show the output frame
            cv2.imshow("Frame", output_color)
            key = cv2.waitKey(10) & 0xFF
            
            # update the FPS counter
            fps.update()
            # if the `Esc` key was pressed, break from the loop
            if key == 27:
                break

    finally:

        # Stop streaming
        pipeline.stop() 
        # do a bit of cleanup
        # stop the timer and save FPS information
        fps.stop()
        
        progress("INFO: elapsed time: {:.2f}".format(fps.elapsed()))
        progress("INFO: approx. FPS: {:.2f}".format(fps.fps()))
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()),file=f)
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()),file=f)
        f.close()    

        progress('INFO:end')
        time.sleep(3)
    
        cv2.destroyAllWindows()
        out.release() 

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------thread 2------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# thread 2

def slide():
    velocity_x = 0 # m/s
    velocity_z = 0
    duration = 5 # s
    while True:
        event.wait()

        if (vehicle.mode.name=='LOITER' or vehicle.mode.name=='AUTO') and vehicle.armed==True:
        
            flight_mode=vehicle.mode.name
            print(flight_mode,file=f)

            vehicle.mode=VehicleMode("GUIDED")
            while not vehicle.mode.name=='GUIDED':
                print('Waiting for mode GUIDED',file=f)
                time.sleep(0.5)
            print(flight_mode,file=f)
            count=0

            while event.is_set() :
                if vehicle.mode.name != 'GUIDED':
                   vehicle.mode=VehicleMode("LAND")
                   while not vehicle.mode.name=='LAND':
                        print('Waiting for mode LAND',file=f)
                        time.sleep(0.5)
                   while vehicle.armed:
                        print('landing',file=f)
                        time.sleep(1)
                   print('landed',file=f)
                   break                 
                print('velocity_y:', str(velocity_y),file=f)
                msg=slide_velocity(velocity_x,velocity_y,velocity_z)
                vehicle.send_mavlink(msg)
                time.sleep(1)
                count +=1
                if count >= 10:
                    vehicle.mode=VehicleMode("LAND")
                    while not vehicle.mode.name=='LAND':
                        print('Waiting for mode LAND',file=f)
                        time.sleep(0.5)
                    while vehicle.armed:
                        print('landing',file=f)
                        time.sleep(1)
                    print('landed',file=f)
                    break
                
            print('move stopped',file=f)
            if vehicle.mode.name != 'LAND':
                vehicle.mode=VehicleMode(flight_mode)
                while not vehicle.mode.name==flight_mode:
                    print('Waiting for mode change ...',file=f)
                    time.sleep(0.5)
                print(flight_mode,file=f)        

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# --------------------------main------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# As of  ardupilot version 4.1 guided commands have changed. In SET_POSITION_TARGET_LOCAL_NED
# one can control position, velocity, acceleration and yaw requiring an appropriate type_mask.
# bit0:PosX, bit1:PosY, bit2:PosZ, bit3:VelX, bit4:VelY, bit5:VelZ, bit6:AccX, bit7:AccY,
# bit8:AccZ, bit10:yaw, bit11:yaw rate. bit9 not used.
# Mavlink Position_Target_TypeMask

X_IGNORE = 1
Y_IGNORE = 2
Z_IGNORE = 4
VX_IGNORE = 8
VY_IGNORE = 16
VZ_IGNORE = 32
AX_IGNORE = 64
AY_IGNORE = 128
AZ_IGNORE = 256
FORCE_SET = 512
YAW_IGNORE = 1024 
YAW_RATE_IGNORE =2048

mask = X_IGNORE | Y_IGNORE | Z_IGNORE | AX_IGNORE | AY_IGNORE | AZ_IGNORE | YAW_RATE_IGNORE # use velocity and Yaw
print('type_mask=',mask)



progress("INFO: Starting Vehicle communications")
# connect vehicle to access point
##vehicle = connect('udpin:192.168.1.61:14550', wait_ready=False)
vehicle = connect('udpin:127.0.0.1:15550', wait_ready=False,source_system=1,source_component=2)
#vehicle = connect('udpout:127.0.0.1:15667', wait_ready=False,source_system=1,source_component=1)
vehicle.initialize(8,30)
vehicle.wait_ready('autopilot_version')
# Get all vehicle attributes (state)
print("\nGet all vehicle attribute values:")
print(" Autopilot Firmware version: %s" % vehicle.version)
print("   Major version number: %s" % vehicle.version.major)
print("   Minor version number: %s" % vehicle.version.minor)
print("   Patch version number: %s" % vehicle.version.patch)
print("   Release type: %s" % vehicle.version.release_type())
print("   Release version: %s" % vehicle.version.release_version())
print("   Stable release?: %s" % vehicle.version.is_stable())
print(" Attitude: %s" % vehicle.attitude)
print(vehicle.attitude.pitch)
print(" Velocity: %s" % vehicle.velocity)
print(" GPS: %s" % vehicle.gps_0)
print(" Flight mode currently: %s" % vehicle.mode.name)

# start threads
f=open('/home/pi/drone_exe/drone/log_real.txt','w')
# f=open('/home/avncalst/Dropbox/rpi/python/log.txt','w')
event = threading.Event()
t1 = threading.Thread(target=ai)
t2 = threading.Thread(target=slide)
t2.daemon=True # has to run in console
t1.start()
t2.start()
t1.join()
        
