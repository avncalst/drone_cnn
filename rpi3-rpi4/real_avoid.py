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
    
#=========================================================================

#=========================================================================
# thread 1

def ai():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    colorizer = rs.colorizer()
    config.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 15)
##    align = rs.align(rs.stream.color)
    dist_thres_cm = 180 # cm
    max_depth_m = 8
    min_depth_m = 0.1

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
            output_image = np.asanyarray(colorizer.colorize(filtered_frame).get_data()) #shape: 136,240,3
            color_image = np.asanyarray(color_frame.get_data())

            # calculate distance
            distances = distances_depth_image(depth_mat,min_depth_m,max_depth_m,depth_scale,corn_1,corn_2)

            # Stack both images horizontally
            output_color = cv2.resize(color_image, (240, 136))
            cv2.rectangle(output_image, corn_1, corn_2, (0, 255, 0), thickness=2)
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


            
            if chan_8 > 1700:
                
                event.clear()
                send_sensor_msg(conn, min_depth_cm=int(min_depth_m*100), max_depth_cm=int(max_depth_m*100), 
                            distance=int(fly_f), orientation=0)
                
            if chan_8 > 1400 and chan_8 < 1700:

                if state == "fly":
                    event.clear()
    
                if state == "avoid":
                    event.set()

                    if label == 'left':
                        velocity_y = 0.8    # go right
                    if label == 'right':
                        velocity_y = -0.8   # go left
                    if label == 'stop':
                        velocity_y = 0
                
            if chan_8 < 1400:
                
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
                   set_mode(conn,b"LAND")
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
f=open('/home/pi/drone_exe/drone/log_real.txt','w')
# f=open('/home/avncalst/Dropbox/rpi/python/log.txt','w')
event = threading.Event()
t1 = threading.Thread(target=ai)
t2 = threading.Thread(target=slide)
t2.daemon=True # has to run in console
t1.start()
t2.start()
t1.join()
        
