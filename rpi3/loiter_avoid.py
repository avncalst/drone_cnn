import threading
from imutils.video.pivideostream import PiVideoStream #settings Pi camera
##from tensorflow.python.platform import gfile
##from keras.preprocessing.image import img_to_array
from imutils.video import FPS
##from keras.models import load_model
from dronekit import connect, VehicleMode
from pymavlink import mavutil 
import imutils
import numpy as np
import time
import cv2
##import tensorflow as tf
from operator import itemgetter
##import VL53L1X

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#=======================choose inference============================
# choose inference: keras (k), tensorflow (tf), opencv.dnn (dnn)
inference='dnn' # tf, k, dnn
#===================================================================
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


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

#=========================================================================
# thread 1

def ai():
    cap=PiVideoStream(resolution=(208,208),framerate=20)    #rpi camera
    cap.start()
    cap.camera.iso=100 # 0:auto, 100-200: sunny day
    cap.camera.awb_mode='sunlight' # sunlight,cloudy,auto
    cap.camera.hflip=True
    cap.camera.vflip=True    
    time.sleep(2.0)

    print("analog_gain: ",float(cap.camera.analog_gain),file=f)
    print("exposure_speed: ",cap.camera.exposure_speed,file=f)
    print("iso: ", cap.camera.iso,file=f)   

    print("[INFO] video recording")
    out = cv2.VideoWriter('avcnet.avi',cv2.VideoWriter_fourcc(*'XVID'), 20, (208,208))


    # load the trained convolutional neural network
    print("[INFO] loading network...")
    print('inference:',inference)
    print('inference:',inference,file=f)

    if inference == 'k':
        from keras.models import load_model
        import tensorflow as tf
        from keras.preprocessing.image import img_to_array
        model = load_model("/home/pi/drone_exe/TF_model/avcnet_best_8.hdf5",
                   custom_objects={"tf": tf} )
    
    if inference == 'tf':
        from tensorflow.python.platform import gfile
        import tensorflow as tf
        from keras.preprocessing.image import img_to_array
        f1 = gfile.FastGFile("/home/pi/drone_exe/TF_model/tf_model_cv.pb", 'rb')
        graph_def = tf.GraphDef()
        # Parses a serialized binary message into the current message.
        graph_def.ParseFromString(f1.read())
        f1.close()
        sess = tf.Session()
        sess.graph.as_default()
        # Import a serialized TensorFlow `GraphDef` protocol buffer
        # and place into the current default `Graph`.
        tf.import_graph_def(graph_def)    
        softmax_tensor = sess.graph.get_tensor_by_name('import/activation_5/Softmax:0')
    
    if inference == 'dnn':
        net=cv2.dnn.readNetFromTensorflow('/home/pi/drone_exe/TF_model/tf_model_cv.pb')
##        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) # if no NCS stick
##        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) # if no NCS stick
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

  
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
##    k=0 # no filtering
    fps = FPS().start()
    
    while True:
        start = time.time()
        frame = cap.read()
        orig  = frame.copy()
##        frame = cv2.resize(frame, (64,64))
##        frame = frame.astype("float")/255.0
##        frame = img_to_array(frame)
##        frame = np.expand_dims(frame, axis=0)

#===========================dnn============================================    
        if inference == 'dnn':
            # use cv2.dnn module for inference
            resized=cv2.resize(frame, (64, 64))
            scale=1/255.0 # AI trained with scaled images image/255
            blob = cv2.dnn.blobFromImage(resized,scale,
                        (64, 64), (0,0,0),swapRB=False,ddepth = 5)
            net.setInput(blob)
            predictions = net.forward()

#===========================tf=============================================        
        if inference == 'tf':
            #use tf for inference
            frame = cv2.resize(frame, (64,64))
            frame = frame.astype("float")/255.0
            frame = img_to_array(frame)
            frame = np.expand_dims(frame, axis=0)    
            predictions = sess.run(softmax_tensor, {'import/img_input:0': frame})

#===========================k==============================================
        if inference == 'k':
            #use k for inference
            frame = cv2.resize(frame, (64,64))
            frame = frame.astype("float")/255.0
            frame = img_to_array(frame)
            frame = np.expand_dims(frame, axis=0)        
            predictions = model.predict(frame)

#===========================================================================

        # classify the input image
        fly = predictions[0][3]
        # build the label
##        my_dict = {'stop':predictions[0][0], 'left':predictions[0][1],
##            'right':predictions[0][2], 'fly':predictions[0][3]}   
        my_dict = {'stop':predictions[0][0], 'left':predictions[0][1],
            'right':predictions[0][2]}        
        maxPair = max(my_dict.items(), key=itemgetter(1))
        fly_f = k*fly_1 + (1-k)*fly
        fly_1 = fly_f


        if state == 'avoid':
            label=maxPair[0]
            proba=maxPair[1]
##            print(label,file=f)
            if fly_f*100 >= 60:
                dist=510
                state='fly'
                print(state,file=f)
                
                
        else:
            label='forward'
            proba=fly_f
##            print(label,proba,file=f)
            if fly_f*100 <= 50:
                dist=180
                label=maxPair[0]
                proba=maxPair[1]
                print(label,file=f)
                state='avoid'
        

        label_1 = "{} {:.1f}% {}".format(label, proba * 100, state)
        # draw the label on the image
        output = imutils.resize(orig, width=208)
        cv2.putText(output, label_1, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

        # Write the frame into the file 'output.avi'
        out.write(output)         

        if vehicle.channels['8'] > 1700:
            event.clear()
        if vehicle.channels['8'] > 1400 and vehicle.channels['8'] < 1700:
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
               
        if vehicle.channels['8'] < 1400:
            event.clear()


        msg_sensor(dist,0,600)        
    
        # show the output frame
        cv2.imshow("Frame", output)
        key = cv2.waitKey(10) & 0xFF
        
        # update the FPS counter
        fps.update()
        # if the `Esc` key was pressed, break from the loop
        if key == 27:
            break


    # do a bit of cleanup
    # stop the timer and save FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()),file=f)
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()),file=f)
    f.close()    
    f.close()
    print('end')
    
    cv2.destroyAllWindows()
    cap.stop()
    out.release()
    
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

#=========================================================================
# thread 3

def range_dist():
    
    sonar=False
    lidar_sens=False # lidar sensing True, else EZ4 sonar or VL53L1X

    if sonar:
        # does not work well on grass
        from smbus import SMBus
##        i2cbus = SMBus(1)
        while True:
            try:
                i2cbus = SMBus(1)
                i2cbus.write_byte(0x70, 0x51)
                time.sleep(0.12)
                val = i2cbus.read_word_data(0x70, 0xE1)
                distance_in_cm=val>>8 | (val & 0x0F)<<8
##                print distance_in_cm, 'cm'
                print(distance_in_cm,'cm',file=f)
                msg_sensor(distance_in_cm,25,400), #sonar facing down
                
            except IOError as err:
                print(err)

            time.sleep(0.1)

    if lidar_sens:
        from lidar_lite import Lidar_Lite
        lidar = Lidar_Lite()
        connected=lidar.connect(1)
        if connected < 0:
            print("\nlidar not connected")
        else:
            print("\nlidar connected")
        while True:
            dist = lidar.getDistance()
            print(dist,'cm')
            print(dist,'cm',file=f)
            msg_sensor(dist,25,700), #lidar facing down
            time.sleep(0.2)
            

    else:
        # does not work in sunlight
        import VL53L1X
        tof = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)
        tof.open() # Initialise the i2c bus and configure the sensor
        tof.start_ranging(2) # Start ranging, 1 = Short Range, 2 = Medium Range, 3 = Long Range
        # Short range max:1.3m, Medium range max:3m, Long range max:4m
        while True:
            distance_in_cm = tof.get_distance()/10 # Grab the range in cm (mm)
            time.sleep(0.2) # timeout mavlink rangefinder = 500 ms
            print(distance_in_cm,'cm',file=f)
            msg_sensor(distance_in_cm,25,250), #sensor facing down

        tof.stop_ranging() # Stop ranging

#=========================================================================
# main

# connect vehicle to access point
##vehicle = connect('udpin:192.168.1.61:14550', wait_ready=False)
vehicle = connect('udpin:127.0.0.1:15550', wait_ready=False)
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
print(" Velocity: %s" % vehicle.velocity)
print(" GPS: %s" % vehicle.gps_0)
print(" Flight mode currently: %s" % vehicle.mode.name)
# start threads
f=open('/home/pi/drone_exe/drone/log.txt','w')
event = threading.Event()
t1 = threading.Thread(target=ai)
t2 = threading.Thread(target=slide)
##t3 = threading.Thread(target=range_dist)
t2.daemon=True # has to run in console
##t3.daemon=True
t1.start()
t2.start()
##t3.start()
t1.join()
        
