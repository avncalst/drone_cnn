import threading
from pymavlink import mavutil 
from numba import njit
import imutils
import numpy as np
import time
import cv2
import pyrealsense2 as rs
import sys
from operator import itemgetter
from imutils.video import FPS
from pymavlink.dialects.v20 import ardupilotmega as mavlink2
# Set MAVLink protocol to 2.
import os
import queue

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# -------------MAVLINK FUNCTIONS------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def send_obstacle_distance_3D_message(conn,obstacle_coordinates,current_time_ms):
    global depth_range,last_obstacle_distance_sent_ms
    if current_time_ms == last_obstacle_distance_sent_ms:
    # no new frame and ends execution  function
        return 
    last_obstacle_distance_sent_ms = current_time_ms
    for i in range(9):
        mesg = mavlink2.MAVLink(None) # Required by MAVLink.__init__, but not needed here
        msg = mesg.obstacle_distance_3d_encode(
                time_boot_ms = current_time_ms,    # ms
                sensor_type = 0,       # not implemented in ArduPilot            
                frame = mavutil.mavlink.MAV_FRAME_BODY_FRD,          
                obstacle_id = 65535,   # unknown ID of the object. We are not really detecting the type of obstacle           
                x = float(obstacle_coordinates[i][0]),	   # X in NEU body frame, in m
                y = float(obstacle_coordinates[i][1]),     # Y in NEU body frame  
                z = float(obstacle_coordinates[i][2]),	   # Z in NEU body frame  
                min_distance = float(depth_range[0]/1000), # min range of sensor, in m
                max_distance = float(depth_range[1]/1000),  # max range of sensor, in m
                )
        # print(msg)
        conn.mav.send(msg)

@njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx

def progress(string):
    print(string, file=sys.stdout)
    sys.stdout.flush()

def staircase(H,i):
    if 0<=i<3 :
        return 0
    if 3<=i<6:
        return H/3
    if 6<=i<9:
        return H/3*2
    if 9<=i<12:
        return H


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

def read_rc_hannel(conn):
    # Read incoming MAVLink messages
    msg = conn.recv_match(type=['RC_CHANNELS'], blocking=True)
    if msg is not None and msg.get_type() == 'RC_CHANNELS':
        # Extract the RC channel 8 value from the message
        return msg.to_dict()  
    

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# -----------------------thread-1-----------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def ai(queue):

    global depth_range, chan_8
    flag_depth = False
    dist_saf = 1 # colorText turns red if distance obstacle < dist_saf
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30) #left IR
    pipeline.start(config)
   
    progress("INFO: video recording")
    out = cv2.VideoWriter('avcnet.avi',cv2.VideoWriter_fourcc(*'XVID'), 20, (640,480))
    
   
    obstacle_coordinates = np.ones((9,3))* 9999
    depth_range = [200,15000] # only points within this interval will be detected with distance !=0
    fps = 0
    startTime = time.monotonic()
    counter = 0
    fontType = cv2.FONT_HERSHEY_TRIPLEX       
    while True:
        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        # ir_left_frame = frames.get_infrared_frame(1)
        # show intrinsic parameters
        # depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        # print(depth_intrin)

        if not depth_frame or not color_frame:
            continue

        depthFrame = np.asanyarray(depth_frame.get_data())
        
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depthFrame, alpha=0.03), cv2.COLORMAP_JET) 
        Frame = np.asanyarray(color_frame.get_data())

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #-----------------detect the closest obstacle in screen divisions----------------------
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The bgr screen has a resolution of 640x480 and is divided into 9 parts. These parts are mapped to
        the depth frame using homograph.py + perspectiveTransform (align_images.py) with topLeft corner offx,offy. 
        The depth frame coincides with the monoLeft cam with optical center pixel coordinates cx,cy. 
        The global X,Y,Z coordinates of the obstacle corresponds to the image u,v coordinates using u = fx.X/Z
        and v = fy.Y/Z. fx anf fy are the focal length along the u anf v axis in pixels. fx and fy depend on the
        image resolution of the monoLeft cam. fx=379 and fy=379 pixels in case of a 640x480 image. Given W 
        of the raster on the monoLeft image the drone will not detect the obstacle provided it is d = W.Z/(2fx)
        away from the optical axis of the monoLeft cam. In case of Z = 2m, d = 0.8m; Z = 3m, d = 1.2m
        The drone uses a FRD local frame attached to the vehicle with
        x: forward, y: right and z: down. This means that the coordinates of the closest obstacle in the 9 parts are 
        converted to the x,y,z of FRD using x: distance d of obstacle, x: (u-cx).d/fx and y: (v-cy).d/fy.

        REMARK: the mapping of the bgr image to the monoLeft cam is based on images at +/- 3m from the vehicle.  
        
        """ 
        # calculate distance
        # coordinates bgr frame
        Wi = 480.0
        Hi = 480.0
        offxi = 80
        offyi = 0
        # coordinates depthframe
        W = 292.0
        H = 295.0
        offx = 166
        offy = 81
        cx = 318
        cy = 209            

        # Set ROI
        for i in range(0,9):
            topLeft = (int(W*(i%3)/3+offx),int(staircase(H,i)+offy))
            bottomRight = (int(W*((i%3)+1)/3)+offx, int(staircase(H,i+3)+offy))
            sub_array = depthFrame[topLeft[1]:bottomRight[1],topLeft[0]:bottomRight[0]]
            inRange = (depth_range[0] <= sub_array) & (sub_array <= depth_range[1])
            result = cv2.minMaxLoc(sub_array[inRange])
            min_idx = index(sub_array,result[0])
            if type(min_idx) is tuple:
                idx = [sum(x) for x in zip(min_idx,topLeft)]
            else:
                idx = topLeft
            d = result[0]/1000.0 # convert distance to m
            
            if flag_depth:
                cv2.rectangle(Frame,topLeft,bottomRight,(0,0,255),3, cv2.FONT_HERSHEY_SIMPLEX)
            
            obstacle_coordinates[i][0] = d
            obstacle_coordinates[i][1] = round((idx[0]-cx)*d/379,3) # x = (u-cx).X/fx
            obstacle_coordinates[i][2] = round((idx[1]-cy)*d/379,3) # y = (v-cy).X/fy 
            euclDist = np.sqrt((obstacle_coordinates[i][0])**2+(obstacle_coordinates[i][1])**2+(obstacle_coordinates[i][2])**2)                  
            topLefti = (int(Wi*(i%3)/3+offxi),int(staircase(Hi,i)+offyi))
            bottomRighti = (int(Wi*((i%3)+1)/3)+offxi, int(staircase(Hi,i+3)+offyi))
            if chan_8 >1500:
                if euclDist > dist_saf:
                    colorText = (255,255,255)   
                else:
                    colorText = (0,0,255)
                cv2.rectangle(Frame,topLefti,bottomRighti,(255,255,255),3, cv2.FONT_HERSHEY_SIMPLEX)
                cv2.putText(Frame,f"d: %5.2f" %euclDist,(topLefti[0] + 10, topLefti[1] + 20), fontType, 0.5, colorText)
        queue.put(obstacle_coordinates)
        # print(obstacle_coordinates)

        cv2.putText(Frame, "NN fps: {:.2f}".format(fps), (2, Frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))
        cv2.imshow("Frame",Frame)
        # cv2.imshow("Depth",depth_colormap)
        # Write the frame into the file 'output.avi'
        out.write(Frame)
    
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break
    out.release()
    pipeline.stop()  


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------thread 2------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def mes(queue):
    global current_time_ms
    start_time =  int(round(time.time() * 1000))
    current_milli_time = lambda: int(round(time.time() * 1000) - start_time)
    obstacle_distance_msg_hz = 15
    mesg = None
    while True:
        current_time_ms = current_milli_time()
        mesg = queue.get()
        send_obstacle_distance_3D_message(conn,mesg,current_time_ms)
        time.sleep(1/obstacle_distance_msg_hz)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------thread 3------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# thread 3
# OA_TYPE 0: no Bendy Ruler, 1: Bendy Ruler active

def chan():
    global chan_8
    chan_12_old = 1200
    param=b'OA_TYPE'# byte string of ASCI characters: O', 'A', '_', and 'T'; b'O\x00A\x00_\x00T\x00'
    read_param(conn,param)
    while True:
        msg = read_rc_hannel(conn)
        chan_8 = msg['chan8_raw']
        chan_12 = msg['chan12_raw']
        if chan_12 > 1500 and chan_12 != chan_12_old:
            # Bendy Ruler on
            set_param(conn,param,1)
            chan_12_old = chan_12
        if chan_12 < 1500 and chan_12 != chan_12_old:
            # Bendy Ruler off
            set_param(conn,param,0)
            chan_12_old = chan_12        

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# --------------------------main------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
progress("INFO: Starting Vehicle communications")

import os
os.environ["MAVLINK20"] = "1"
last_obstacle_distance_sent_ms = 0
global chan_8

flag_drone = False

if flag_drone:
    IP = 'udpin:127.0.0.1:15550'
else:
    IP = 'udpin:192.168.1.32:15550'
    # chan_8 = 1800

conn = mavutil.mavlink_connection(
    device=IP,
    autoreconnect = True,
    source_system = 1,
    source_component = 93,
    # /baud=57600,
    force_connected=True,
   )

wait_conn() # send a ping to start connection @ udpin starts sending info
progress("INFO: vehicle connected")

# set_mode(conn,"LOITER")
# print("mode",read_mode(conn))

# start threads
queue = queue.Queue()
event = threading.Event()
t1 = threading.Thread(target=ai,args=(queue,))
t2 = threading.Thread(target=mes,args=(queue,))
t3 = threading.Thread(target=chan)
t2.daemon=True # has to run in console
t3.daemon=True
t1.start()
t2.start()
t3.start()
t1.join() # ends when task finished
        
