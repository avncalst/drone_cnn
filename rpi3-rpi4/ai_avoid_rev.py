import threading
from pymavlink import mavutil 
import imutils
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import cv2
import pyrealsense2 as rs
import sys
from operator import itemgetter
from imutils.video import FPS

# Set MAVLink protocol to 2.
import os
os.environ["MAVLINK20"] = "1"

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# -------------MAVLINK FUNCTIONS------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#=========================================================================
def slide_velocity(conn, velocity_x, velocity_y, velocity_z):
    conn.mav.set_position_target_local_ned_send(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_NED, # frame Needs to be MAV_FRAME_BODY_NED for forward/back left/right control.
        0b0000111111000111, # type_mask
        0, 0, 0, # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z, # m/s
        0, 0, 0, # x, y, z acceleration
        0, 0
        )

def read_param(conn, param):
    # conn.wait_heartbeat()
    conn.mav.param_request_read_send(
        0, 0,
        param,
    -1
    )
    # Print old parameter value
    message = conn.recv_match(type='PARAM_VALUE', blocking=True).to_dict()
    print('name: %s\tvalue: %d' %
      (message['param_id'], message['param_value']))

    time.sleep(1)

def set_param(conn,param,value):
    # conn.wait_heartbeat()
    conn.mav.param_set_send(
        0, 0,
        param,
        value,
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )
    message = conn.recv_match(type='PARAM_VALUE', blocking=True).to_dict()
    print('name: %s\tvalue: %d' %
      (message['param_id'], message['param_value']))

    time.sleep(1)

def read_mode(conn):
    # reliable reading of flightmode
    msg=None
    while msg==None:
        msg = conn.recv_match(type = 'HEARTBEAT', blocking = False)
        # print('msg:',msg)
        if msg:
            mode = mavutil.mode_string_v10(msg)
            return mode


def set_mode(conn,mode):
    # Get mode ID, mode string, ex:'LOITER'
    mode_id = conn.mode_mapping()[mode]
    # Set new mode
    # conn.mav.command_long_send(
    #    master.target_system, master.target_component,
    #    mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
    #    0, mode_id, 0, 0, 0, 0, 0)
    # or:
    # conn.set_mode(mode_id)
    # or:
    conn.mav.set_mode_send(
        0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id)

    while True:
        # Wait for ACK command
        ack_msg = conn.recv_match(type='COMMAND_ACK', blocking=True)
        ack_msg = ack_msg.to_dict()

        # Check if command in the same in `set_mode`
        if ack_msg['command'] != mavutil.mavlink.MAVLINK_MSG_ID_SET_MODE:
            continue

        # Print the ACK result !
        print('ACK result ok')
        # print(mavutil.mavlink.enums['MAV_RESULT'][ack_msg['result']].description)
        break

def send_sensor_msg(conn, min_depth_cm, max_depth_cm, distance, orientation):
    # Average out a portion of the centermost part
    conn.mav.distance_sensor_send(
        0,                  # ms Timestamp (UNIX time or time since system boot) (ignored)
        min_depth_cm,       # min_distance, uint16_t, cm
        max_depth_cm,       # min_distance, uint16_t, cm
        distance,           # current_distance,	uint16_t, cm	
        0,	                # type : 0 (ignored)
        0,                  # id : 0 (ignored)
        orientation,        # orientation: 0...7, 25
        0                   # covariance : 0 (ignored)
    )
    
def wait_conn():
    """
    Sends a ping to estabilish the UDP communication and awaits for a response
    """
    msg = None
    while not msg:
        conn.mav.ping_send(
            int(time.time() * 1e6), # Unix time in microseconds
            0, # Ping number
            0, # Request ping of all systems
            0 # Request ping of all components
        )
        msg = conn.recv_match()
        time.sleep(0.5)

def progress(string):
    print(string, file=sys.stdout)
    sys.stdout.flush()
    
#=========================================================================

#=========================================================================
# thread 1

def ai():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 15)

    progress("INFO: video recording")
    out = cv2.VideoWriter('drone.avi',cv2.VideoWriter_fourcc(*'XVID'), 15, (240,136))
    
    progress("INFO: load inference")
    #=======================================================================
    # #-----------------DNN inference-----------------------------------------
    # net=cv2.dnn.readNetFromTensorflow('/home/pi/drone_exe/TF_model/tf_model_cv.pb')
    # # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) # if no NCS stick
    # # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) # if no NCS stick
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

    #=======================================================================
    #------------------TFLITE inference-------------------------------------
    interpreter = tflite.Interpreter('/home/pi/drone_exe/TF_model/w_tflite_model_edgetpu.tflite',
            experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.allocate_tensors()
    #=======================================================================
  
    # default parameters
    orient=0
    distmax=600
    tim_old=0.1
    state='fly'
    dist=510
    global velocity_y
    velocity_y=0
    fly_1=0.9
    k=0.66 # k=(tau/T)/(1+tau/T) tau time constant LPF, T period
    # k=0 # no filtering
    fps = FPS().start()
    # Start streaming
    pipeline.start(config)
    
    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

          
            color_image = np.asanyarray(color_frame.get_data())
            #=================DNN==============================
            # frame = color_image[:,92:332] # square image 424x424
            # # print("shape:",frame.shape)
            # orig  = frame.copy()
            # # use cv2.dnn module for inference
            # resized=cv2.resize(frame, (64, 64))
            # scale=1/255.0 # AI trained with scaled images image/255
            # blob = cv2.dnn.blobFromImage(resized,scale,
            #             (64, 64), (0,0,0),swapRB=False,ddepth = 5)
            # net.setInput(blob)
            # predictions = net.forward()
            # fly = predictions[0][3]
            # my_dict = {'stop':predictions[0][0], 'left':predictions[0][1],
            #     'right':predictions[0][2]}              
            #================TFLITE=============================
            orig = color_image.copy()
            output = cv2.resize(orig,(240,136))
            frame=cv2.resize(color_image, (224, 224))
            cv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) # RGB colors needed
            frame = frame/255.0 # image is normalized
            frame = np.expand_dims(frame, axis=0)
            frame = frame.astype('float32')  # float32

            interpreter.set_tensor(input_details[0]['index'], frame)
            interpreter.invoke()
            output_details = interpreter.get_output_details()[0]
            fly = interpreter.tensor(output_details['index'])().flatten()[0]
            left = interpreter.tensor(output_details['index'])().flatten()[1]
            right = interpreter.tensor(output_details['index'])().flatten()[2]
            stop = interpreter.tensor(output_details['index'])().flatten()[3]
            my_dict = {'stop':stop, 'left':left, 'right':right} 
            #=======================================================
            maxPair = max(my_dict.items(), key=itemgetter(1))
            fly_f = k*fly_1 + (1-k)*fly
            fly_1 = fly_f

            if state == 'avoid':
    ##            label=maxPair[0]
    ##            proba=maxPair[1]
                if fly_f*100 >= 60:
                    dist=510
                    state='fly'
                    print(state,file=f)
                
            else:
                label='forward'
                proba=fly
                if fly_f*100 <= 50:
                    dist=180
                    label=maxPair[0]
                    proba=maxPair[1]
                    print(my_dict,file=f)
                    state='avoid'

            label_1 = "{} {:.1f}% {}".format(label, proba * 100, state)
            # draw the label on the image
##            output = imutils.resize(orig, width=208)
            cv2.putText(output, label_1, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
            # Write the frame into the file 'drone.avi'
            out.write(output)


            chan_8 = conn.messages['RC_CHANNELS_RAW'].chan8_raw
            if chan_8 > 1700:
                event.clear()
            if chan_8 > 1400 and chan_8 < 1700:
                if state == "fly":
                    event.clear()
    
                if state == "avoid":
                    event.set()

                    if label == 'left':
                        velocity_y = -0.8
                    if label == 'right':
                        velocity_y = 0.8
                    if label == 'stop':
                        velocity_y = 0
                
            if chan_8 < 1400:
                event.clear()

            send_sensor_msg(conn, min_depth_cm=10, max_depth_cm=600, distance=dist, orientation=0)       
        
            # show the output frame
            cv2.imshow("Frame", output)
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
    
        cv2.destroyAllWindows()
        out.release()   
#=========================================================================
# thread 2

def slide():
    velocity_x = 0 # m/s
    velocity_z = 0
    duration = 5 # s
    while True:
        event.wait()
        if (read_mode(conn)=='LOITER' or read_mode(conn)=='AUTO') and bool(conn.motors_armed())==True:
        # if (read_mode(conn)=='LOITER' or read_mode(conn)=='AUTO') ==True:
        
            flight_mode=read_mode(conn)
            print(flight_mode,file=f)

            set_mode(conn,"GUIDED")
            while not read_mode(conn)=='GUIDED':
                print('Waiting for mode GUIDED',file=f)
                time.sleep(0.5)
            print(flight_mode,file=f)
            count=0

            while event.is_set() :
                if read_mode(conn) != 'GUIDED':
                   set_mode(conn,"LAND")
                   while not read_mode(conn)=='LAND':
                        print('Waiting for mode LAND',file=f)
                        time.sleep(0.5)
                   while bool(conn.motors_armed()):
                        print('landing',file=f)
                        time.sleep(1)
                   print('landed',file=f)
                   break                 
                print('velocity_y:', str(velocity_y),file=f)
                slide_velocity(conn,velocity_x,velocity_y,velocity_z)
                time.sleep(1)
                count +=1
                if count >= 10:
                    set_mode(conn,"LAND")
                    while not read_mode(conn)=='LAND':
                        print('Waiting for mode LAND',file=f)
                        time.sleep(0.5)
                    while bool(conn.motors_armed()):
                        print('landing',file=f)
                        time.sleep(1)
                    print('landed',file=f)
                    break
                
            print('move stopped',file=f)
            if read_mode(conn) != 'LAND':
                set_mode(conn,flight_mode)
                while not read_mode(conn)==flight_mode:
                    print('Waiting for mode change ...',file=f)
                    time.sleep(0.5)
                print(flight_mode,file=f)

#=========================================================================
# main

progress("INFO: Starting Vehicle communications")

conn = mavutil.mavlink_connection(
    device='udpin:127.0.0.1:15550',
    autoreconnect = True,
    source_system = 1,
    source_component = 93,
    baud=57600,
    force_connected=True,
)
wait_conn() # send a ping to start connection @ udpin starts sending info
progress("INFO: vehicle connected")


# Get vehicle attributes (state)
param=b'AVOID_ENABLE'
read_param(conn,param)
set_mode(conn,"LOITER")
print("mode",read_mode(conn))
chan_8 = conn.messages['RC_CHANNELS_RAW'].chan8_raw
progress("INFO: chan_8 %s"%chan_8)

if chan_8 > 1700:
    param=b'AVOID_ENABLE'
    set_param(conn,param,2)
    read_param(conn,param)
    param=b'PRX_TYPE'
    set_param(conn,param,2)
    read_param(conn,param)
   
if chan_8 > 1400 and chan_8 < 1700:
    param=b'AVOID_ENABLE'
    set_param(conn,param,0)
    read_param(conn,param)
    param=b'PRX_TYPE'
    set_param(conn,param,0)
    read_param(conn,param)
       
if chan_8 < 1400:
    param=b'AVOID_ENABLE'
    set_param(conn,param,0)
    read_param(conn,param)
    param=b'PRX_TYPE'
    set_param(conn,param,0)
    read_param(conn,param)

# start threads
f=open('/home/pi/drone_exe/drone/log.txt','w')
# f=open('/home/avncalst/Dropbox/rpi/python/log.txt','w')
event = threading.Event()
t1 = threading.Thread(target=ai)
t2 = threading.Thread(target=slide)
t2.daemon=True # has to run in console
t1.start()
t2.start()
t1.join()
        
