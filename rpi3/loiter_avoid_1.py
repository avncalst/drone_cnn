import threading
from imutils.video.pivideostream import PiVideoStream #settings Pi camera
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from dronekit import connect, VehicleMode
from pymavlink import mavutil 
import imutils
import numpy as np
import time
import cv2
import tensorflow as tf
from operator import itemgetter


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

def msg_sensor(dist,orient):
    
    msg = vehicle.message_factory.distance_sensor_encode(
            0,      # time system boot, not used
            20,     # min disance cm
            600,    # max dist cm
            dist,   # current dist, int
            0,      # type sensor
            1,      # on board id, not used
            orient, # orientation: 0...7
            0,      # covariance, not used
            )
    
    vehicle.send_mavlink(msg)
#=========================================================================

#=========================================================================
# thread 1

def ai():
    cap=PiVideoStream(resolution=(208,208))                  #rpi camera
    cap.start()
    time.sleep(2.0)
    cap.camera.hflip=True
    cap.camera.vflip=True

    print("[INFO] video recording")
##    out = cv2.VideoWriter('avcnet.avi',cv2.VideoWriter_fourcc(*'XVID'), 10, (208,208))

    # load the trained convolutional neural network
    print("[INFO] loading network...")
##    model = load_model("./avcnet_v1.model")
    model = load_model("./avcnet_best_1.hdf5",custom_objects={"tf": tf} )
  
    # default parameters
    orient=0
    tim_old=0.1
    state='fly'
    dist=510
    global velocity_y
    velocity_y=0

    while True:
        start = time.time()
        frame = cap.read()
        orig  = frame.copy()
        frame = cv2.resize(frame, (64,64))
        frame = frame.astype("float")/255.0
        frame = img_to_array(frame)
        frame = np.expand_dims(frame, axis=0)

        

        # classify the input image
        (stop, left,right,fly) = model.predict(frame)[0]
        # build the label
##        my_dict = {'stop':stop, 'left':left, 'right':right,'fly':fly}
        my_dict = {'stop':stop, 'left':left, 'right':right}        
        maxPair = max(my_dict.iteritems(), key=itemgetter(1))


        if state == 'avoid':
##            label=maxPair[0]
##            proba=maxPair[1]
            if fly*100 >= 60:
                dist=510
                state='fly'
                print >>f,state
                
                
        else:
            label='forward'
            proba=fly
            if fly*100 <= 50:
                dist=180
                label=maxPair[0]
                proba=maxPair[1]
                print >>f,my_dict
                state='avoid'
        

        label_1 = "{} {:.1f}% {}".format(label, proba * 100, state)
        # draw the label on the image
        output = imutils.resize(orig, width=208)
        cv2.putText(output, label_1, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

        # Write the frame into the file 'output.avi'
##        out.write(output)         

        if vehicle.channels['8'] > 1700:
            event.clear()
        if vehicle.channels['8'] > 1400 and vehicle.channels['8'] < 1700:
            if state == "fly":
                event.clear()
 
            if state == "avoid":
                event.set()

                if label == 'left':
                    velocity_y = -0.5
                if label == 'right':
                    velocity_y = 0.5
                if label == 'stop':
                    velocity_y = 0
               
        if vehicle.channels['8'] < 1400:
            event.clear()


        msg_sensor(dist,orient)        
    
        # show the output frame
        cv2.imshow("Frame", output)
        key = cv2.waitKey(10) & 0xFF

        # if the `Esc` key was pressed, break from the loop
        if key == 27:
            break


    # do a bit of cleanup
    f.close()
    print('end')
    
    cv2.destroyAllWindows()
    cap.stop()
##    out.release()    
#=========================================================================
# thread 2

def slide():
    velocity_x = 0 # m/s
    velocity_z = 0
    duration = 5 # s
    while True:
        event.wait()
        if (vehicle.mode.name=='LOITER' or vehicle.mode.name=='AUTO') and vehicle.armed==True:
        
            flight_mode=vehicle.mode.name
            print >>f, flight_mode

            vehicle.mode=VehicleMode("GUIDED")
            while not vehicle.mode.name=='GUIDED':
                print >>f, 'Waiting for mode GUIDED'
                time.sleep(0.5)
            print >>f, flight_mode
            count=0

            while event.is_set() :
                if vehicle.mode.name != 'GUIDED':
                   vehicle.mode=VehicleMode("LAND")
                   while not vehicle.mode.name=='LAND':
                        print >>f, 'Waiting for mode LAND'
                        time.sleep(0.5)
                   while vehicle.armed:
                        print >>f, 'landing'
                        time.sleep(1)
                   print >>f, 'landed'
                   break                 
                print >>f,'velocity_y:', str(velocity_y)
                msg=slide_velocity(velocity_x,velocity_y,velocity_z)
                vehicle.send_mavlink(msg)
                time.sleep(1)
                count +=1
                if count >= 10:
                    vehicle.mode=VehicleMode("LAND")
                    while not vehicle.mode.name=='LAND':
                        print >>f, 'Waiting for mode LAND'
                        time.sleep(0.5)
                    while vehicle.armed:
                        print >>f, 'landing'
                        time.sleep(1)
                    print >>f, 'landed'
                    break
                
            print >>f, 'move stopped'
            if vehicle.mode.name != 'LAND':
                vehicle.mode=VehicleMode(flight_mode)
                while not vehicle.mode.name==flight_mode:
                    print >>f, 'Waiting for mode change ...'
                    time.sleep(0.5)
                print >>f, flight_mode


#=========================================================================
# main

# connect vehicle to access point
##vehicle = connect('udpin:192.168.1.61:14550', wait_ready=False)
vehicle = connect('udpin:127.0.0.1:15550', wait_ready=False)
vehicle.initialize(8,30)
vehicle.wait_ready('autopilot_version')
# Get all vehicle attributes (state)
print "\nGet all vehicle attribute values:"
print " Autopilot Firmware version: %s" % vehicle.version
print "   Major version number: %s" % vehicle.version.major
print "   Minor version number: %s" % vehicle.version.minor
print "   Patch version number: %s" % vehicle.version.patch
print "   Release type: %s" % vehicle.version.release_type()
print "   Release version: %s" % vehicle.version.release_version()
print "   Stable release?: %s" % vehicle.version.is_stable()
print " Attitude: %s" % vehicle.attitude
print " Velocity: %s" % vehicle.velocity
print " GPS: %s" % vehicle.gps_0
print " Flight mode currently: %s" % vehicle.mode.name
# start threads
f=open('log.txt','w')
event = threading.Event()
t1 = threading.Thread(target=ai)
t2 = threading.Thread(target=slide)
t2.daemon=True # has to run in console
t1.start()
t2.start()
t1.join()
        
