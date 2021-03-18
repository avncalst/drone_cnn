#!/usr/bin/env python

import sys
import os
# sys.path.append('/usr/lib/python3/dist-packages')
# print(sys.path)



import rospy
import cv2
import imutils
import threading
import time
from pymavlink import mavutil


import numpy as np
from sensor_msgs.msg            import Image
from cv_bridge                  import CvBridge, CvBridgeError
from operator                   import itemgetter
from pymavlink                  import mavutil

# Set MAVLink protocol to 2.
import os
os.environ["MAVLINK20"] = "1"

"""
ON PC: roslaunch roslaunch gazebo_ros empty_world.launch
   0------------------> x (cols) Image Frame
   |
   |        c    Camera frame
   |         o---> x
   |         |
   |         V y
   |
   V y (rows)
SUBSCRIBES TO:
    /iris/camera/image_raw: Source image topic
    
PUBLISHES TO:

"""

velocity_y=0.0
k = 0.66
fly_1=0.9
state='fly'
label='forward'
proba=1
dist=510

# use cv2.dnn module for inference
net=cv2.dnn.readNetFromTensorflow('/home/avncalst/Dropbox/donkeycar/mycar/models/tf_model_cv.pb')
# net=cv2.dnn.readNetFromTensorflow('/home/avncalst/Dropbox/donkeycar/mycar/models/sitl.pb')
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) # if no NCS stick
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) # if no NCS stick

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



#=========================================================================

def callback(data):
    #--- Assuming image is 208x208
    bridge = CvBridge()
    global k
    global fly_1
    global state
    global label
    global proba
    global velocity_y
    global dist
    global net


   
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
        
    resized=cv2.resize(cv_image, (64, 64))
    scale=1/255.0 # AI trained with scaled images image/255
    blob = cv2.dnn.blobFromImage(resized,scale,
                (64, 64), (0,0,0),swapRB=False,ddepth = 5)
    net.setInput(blob)
    predictions = net.forward()
    # print(predictions)

    #===========================================================================
    # classify the input image
    
    fly = predictions[0][3]
    # print('fly:',fly)
    # build the label
    my_dict = {'stop':predictions[0][0], 'left':predictions[0][1],
                'right':predictions[0][2]}
            
    maxPair = max(my_dict.items(), key=itemgetter(1))
    fly_f = k*fly_1 + (1-k)*fly
    fly_1 = fly_f


    if state == 'avoid':
        # label=maxPair[0]
        # proba=maxPair[1] 
        if fly_f*100 >= 60:
            dist=510
            state='fly'
   
            
    else:
        label='forward'
        proba=fly
        if fly_f*100 <= 50:
            dist=180
            label=maxPair[0]
            proba=maxPair[1]
            state='avoid'
    

    label_1 = "{} {:.1f}% {}".format(label, proba * 100, state)
    # draw the label on the image
    output = imutils.resize(cv_image, width=208)
    cv2.putText(output, label_1, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)

    if state == "fly":
        event.clear()
        
 
    if state == "avoid":
        event.set()
        # print('avoid-event.set')
        if label == 'left':
            velocity_y = -1
        if label == 'right':
            velocity_y = 1
        if label == 'stop':
            velocity_y = 0.0

    send_sensor_msg(conn, min_depth_cm=10, max_depth_cm=600, distance=510, orientation=0) # prevent flight controller to move copter backwards
    
        
    
 

    # show the output frame
    cv2.imshow("Frame", output)
    key = cv2.waitKey(10) & 0xFF
 
    # # if the `Esc` key was pressed, destroy windows
    # if key == 27:
    #     cv2.destroyAllWindows()

def subscriber():
    rospy.Subscriber("/iris/camera/image_raw", Image, callback)
    print ("<< Subscribed to topic /iris/camera/image_raw")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("Shutting down")

#=========================================================================
# thread 2

def slide():
    velocity_x = 0  # m/s
    velocity_z = 0
    duration = 5  # s
    while True:
        event.wait()
        if (read_mode(conn) == 'LOITER' or read_mode(conn) == 'AUTO') and bool(conn.motors_armed()) == True:
            # if (read_mode(conn)=='LOITER' or read_mode(conn)=='AUTO') ==True:

            flight_mode = read_mode(conn)
            print("flight_mode:",flight_mode)

            set_mode(conn, "GUIDED")
            while not read_mode(conn) == 'GUIDED':
                print('Waiting for mode GUIDED')
                time.sleep(0.5)
            print("flight_mode:",flight_mode)
            count = 0

            while event.is_set():
                if read_mode(conn) != 'GUIDED':
                    set_mode(conn, "LAND")
                    while not read_mode(conn) == 'LAND':
                        print('Waiting for mode LAND')
                        time.sleep(0.5)
                    while bool(conn.motors_armed()):
                        print('landing')
                        time.sleep(1)
                    print('landed')
                    break
                print('velocity_y:', str(velocity_y))
                slide_velocity(conn, velocity_x, velocity_y, velocity_z)
                time.sleep(1)
                count += 1
                if count >= 10:
                    set_mode(conn, "LAND")
                    while not read_mode(conn) == 'LAND':
                        print('Waiting for mode LAND')
                        time.sleep(0.5)
                    while bool(conn.motors_armed()):
                        print('landing')
                        time.sleep(1)
                    print('landed')
                    break

            print('move stopped')
            if read_mode(conn) != 'LAND':
                set_mode(conn, flight_mode)
                while not read_mode(conn) == flight_mode:
                    print('Waiting for mode change ...')
                    time.sleep(0.5)
                print("flight_mode:",flight_mode)

#=========================================================================
        

if __name__ == "__main__":

    print("INFO: Starting Vehicle communications")

    conn = mavutil.mavlink_connection(
        device='udpin:127.0.0.1:14550',
        autoreconnect=True,
        source_system=1,
        source_component=93,
        baud=57600,
        force_connected=True,
    )
    wait_conn()  # send a ping to start connection @ udpin starts sending info
    print("INFO: vehicle connected")


    # Get vehicle attributes (state)
    param = b'AVOID_ENABLE'
    read_param(conn, param)
    set_mode(conn, "LOITER")
    print("mode", read_mode(conn))
       

    # start threads
 
    event = threading.Event()
    t2 = threading.Thread(target=slide)
    t2.daemon = True  # has to run in console
    # t1.start()
    t2.start()
    # t1.join()
    
    rospy.init_node("avctech")
    subscriber()



