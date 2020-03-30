#!/usr/bin/env python

import rospy
import cv2
import imutils
import threading
import time

from sensor_msgs.msg        import Image
from cv_bridge              import CvBridge, CvBridgeError
from operator               import itemgetter
from dronekit               import connect, VehicleMode
from pymavlink              import mavutil 


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
net=cv2.dnn.readNetFromTensorflow('/home/avncalst/Dropbox/donkeycar/mycar/models/sitl.pb')
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) # if no NCS stick
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) # if no NCS stick

#=========================================================================
def slide_velocity(velocity_x, velocity_y, velocity_z):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_NED, # frame Needs to be MAV_FRAME_BODY_NED for forward/back left/right control.
        0b0000111111000111, # type_mask
        0, 0, 0, # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z, # m/s
        0, 0, 0, # x, y, z acceleration
        0, 0)
    return msg
##    for x in range(0,duration):
##        vehicle.send_mavlink(msg)
##        time.sleep(1)


def set_speed(speed):
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, #command
        0, #confirmation
        0, #speed type, ignore on ArduCopter
        speed, # speed
        0, 0, 0, 0, 0 #ignore other parameters
        )
    vehicle.send_mavlink(msg)
    # return msg



def condition_yaw(heading, relative=True):
    """
    Send MAV_CMD_CONDITION_YAW message to point vehicle at a specified heading (in degrees).
    This method sets an absolute heading by default, but you can set the `relative` parameter
    to `True` to set yaw relative to the current yaw heading.
    By default the yaw of the vehicle will follow the direction of travel. After setting 
    the yaw using this function there is no way to return to the default yaw "follow direction 
    of travel" behaviour (https://github.com/diydrones/ardupilot/issues/2427)
    For more information see: 
    http://copter.ardupilot.com/wiki/common-mavlink-mission-command-messages-mav_cmd/#mav_cmd_condition_yaw
    """
    if relative:
        is_relative = 1 #yaw relative to direction of travel
    else:
        is_relative = 0 #yaw is an absolute angle
    # create the CONDITION_YAW command using command_long_encode()
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_CONDITION_YAW, #command
        0, #confirmation
        heading,    # param 1, yaw in degrees
        0,          # param 2, yaw speed deg/s
        1,          # param 3, direction -1 ccw, 1 cw
        is_relative, # param 4, relative offset 1, absolute angle 0
        0, 0, 0)    # param 5 ~ 7 not used
    # send command to vehicle
    vehicle.send_mavlink(msg)

def msg_sensor(dist,orient,distmax):
    
    msg = vehicle.message_factory.distance_sensor_encode(
            0,      # time system boot, not used
            10,     # min disance cm
            distmax,# max dist cm
            dist,   # current dist, int
            0,      # type sensor, laser
            1,      # on board id, not used
            orient, # orientation: 0...7, 25
            0,      # covariance, not used
            )
    
    vehicle.send_mavlink(msg)

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
    # # use cv2.dnn module for inference
    # net=cv2.dnn.readNetFromTensorflow('/home/avncalst/Dropbox/donkeycar/mycar/models/sitl.pb')
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) # if no NCS stick
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) # if no NCS stick
    
    resized=cv2.resize(cv_image, (64, 64))
    scale=1/255.0 # AI trained with scaled images image/255
    blob = cv2.dnn.blobFromImage(resized,scale,
                (64, 64), (0,0,0),swapRB=False,ddepth = 5)
    net.setInput(blob)
    predictions = net.forward()
    # print(predictions)
    
    #===========================================================================

    # classify the input image
    fly = predictions[0][0]
    # build the label

    my_dict = {'stop':predictions[0][3], 'left':predictions[0][1],
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
    # print(label,proba*100,state)
    cv2.putText(output, label_1, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)

    if state == "fly":
        event.clear()
        
 
    if state == "avoid":
        event.set()

        if label == 'left':
            velocity_y = -0.8
        if label == 'right':
            velocity_y = 0.8
        if label == 'stop':
            velocity_y = 0.0

    msg_sensor(510,0,600) # prevent flight controller to move copter backwards
    
        
    
 

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
    velocity_x = 0.0 # m/s
    velocity_z = 0.0
    duration = 5 # s

 
    while True:
        event.wait()
    
        if (vehicle.mode.name=='LOITER' or vehicle.mode.name=='AUTO') and vehicle.armed==True:
        
            flight_mode=vehicle.mode.name
            print(flight_mode)

            vehicle.mode=VehicleMode("GUIDED")
            while not vehicle.mode.name=='GUIDED':
                print('Waiting for mode GUIDED')
                time.sleep(0.5)

            count=0

            while event.is_set() :
                if vehicle.mode.name != 'GUIDED':
                    vehicle.mode=VehicleMode("LAND")
                    while not vehicle.mode.name=='LAND':
                        print('Waiting for mode LAND')
                        time.sleep(0.5)
                    while vehicle.armed:
                        print('landing')
                        time.sleep(1)
                    print('landed')
                    break
                


                print('velocity_y:', count, velocity_y)
                msg=slide_velocity(velocity_x,velocity_y,velocity_z)
                vehicle.send_mavlink(msg)
                time.sleep(1)
                count +=1
                if count >= 40:
                    vehicle.mode=VehicleMode("LAND")
                    while not vehicle.mode.name=='LAND':
                        print('Waiting for mode LAND')
                        time.sleep(0.5)
                    while vehicle.armed:
                        print('landing')
                        time.sleep(1)
                    print('landed')
                    break
                
            print('move stopped')
            if vehicle.mode.name != 'LAND':
                vehicle.mode=VehicleMode(flight_mode)
                while not vehicle.mode.name==flight_mode:
                    print('Waiting for mode change ...')
                    time.sleep(0.5)
                print(flight_mode)

#=========================================================================
        

if __name__ == "__main__":
    #=========================================================================
    # main

    # connect vehicle to access point
    ##vehicle = connect('udpin:192.168.1.61:14550', wait_ready=False)
    vehicle = connect('udpin:127.0.0.1:14550', wait_ready=False)
    vehicle.initialize(8,30)
    vehicle.wait_ready('autopilot_version')
    # Set avoidance enable, proximity type mavlink
    vehicle.parameters['avoid_enable']=2
    vehicle.parameters['PRX_TYPE']=2
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
    print(" Velocity: %s" % vehicle.velocity)
    print(" GPS: %s" % vehicle.gps_0)
    print(" Flight mode currently: %s" % vehicle.mode.name)
    # parameter 0: not used
    print('avoid_enable: %s' % vehicle.parameters['avoid_enable'])
    print('proximity_type: %s' % vehicle.parameters['PRX_TYPE'])



    # start threads
    event = threading.Event()
    # t1 = threading.Thread(target=ai)
    # thread launched at subscriber callback function
    t2 = threading.Thread(target=slide)
    t2.daemon=True # has to run in console
    # t1.start()
    t2.start()
    # t2.join()
    rospy.init_node("avctech")
    subscriber()



