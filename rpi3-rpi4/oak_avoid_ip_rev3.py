##!/usr/bin/env python3

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# code inspired on: https://github.com/rishabsingh3003/ardupilot_depthai_scripts
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import threading
import cv2
from simple_pid import PID
import depthai as dai
import numpy as np
import time
import sys
from pymavlink import mavutil
from numba import njit
from pymavlink.dialects.v20 import ardupilotmega as mavlink2
import queue


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ---------------------functions------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def send_obstacle_distance_3D_message(conn,obstacle_coordinates,current_time_ms):
    global depth_range,last_obstacle_distance_sent_ms
    if current_time_ms == last_obstacle_distance_sent_ms:
    # no new frame and ends execution  function
        return 

    last_obstacle_distance_sent_ms = current_time_ms
    mesg = mavlink2.MAVLink(None) # Required by MAVLink.__init__, but not needed here
    for i in range(3):
        # current_time_ms = int(round(time.time() * 1000))
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
        conn.mav.send(msg)
    # print(msg)
    # print(obstacle_coordinates)        
       

@njit
def obstac(arr,obs_coord,cu,cv,row,topLeft,depth_range):
    h = arr.shape[0] # rows
    w = arr.shape[1] # columns
    for j in range(0,h,10):      
        for k in range(0,w,10):
            # print(arr[j,k])
            # d = round(arr[j,k]/1000,2)
            d = arr[j,k]/1000
            idx = [sum(x) for x in zip((k,j),topLeft)]
            obs_temp = np.array([[d, (idx[0]-cu)*d/454,  (idx[1]-cv)*d/504]])
            dist_t = np.linalg.norm(obs_temp)
            dist = np.linalg.norm(obs_coord[row]) # select row
            if dist_t !=0 and dist_t < dist:
                obs_coord[row] = obs_temp # select row

    if  depth_range[1] < dist :
        obs_coord[row] = np.array([[0,0,0]])
        dist = 0
    return dist,obs_coord[row]


def progress(string):
    print(string, file=sys.stdout)
    sys.stdout.flush()

  
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
        
def read_rc_hannel(conn):
    # Read incoming MAVLink messages
    msg = conn.recv_match(type=['RC_CHANNELS'], blocking=True)
    if msg is not None and msg.get_type() == 'RC_CHANNELS':
        # Extract the RC channel 8 value from the message
        return msg.to_dict()       

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

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# -----------------------thread-1-----------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def ai(queue2):

    global depth_range, chan_8, obstacle_coordinates, current_time_ms
    flag_depth = False
    dist_saf = 1 # colorText turns red if distance obstacle < dist_saf
    depth_range = (400,5000) # only points within this interval will be detected with distance !=0


    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.createColorCamera()
    monoLeft = pipeline.createMonoCamera()
    monoRight = pipeline.createMonoCamera()
    stereoDepth = pipeline.createStereoDepth()

    xoutRgb = pipeline.createXLinkOut()
    xoutDepth = pipeline.createXLinkOut()
    xoutRgb.setStreamName("rgb")
    xoutDepth.setStreamName("depth")
    
    # Properties
    camRgb.setPreviewSize(640, 400)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    # camRgb.setFps(30)
    camRgb.setFps(20)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

 
    # On-Device Post-Processing Filters Configuration ---
    config = stereoDepth.initialConfig.get()

    # Median Filter (Hardware accelerated, fast noise reduction)
    # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
    stereoDepth.initialConfig.setConfidenceThreshold(130)
    stereoDepth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereoDepth.setSubpixel(True)
    stereoDepth.setLeftRightCheck(True)

    config = stereoDepth.initialConfig.get()
    config.postProcessing.speckleFilter.enable = False
    config.postProcessing.speckleFilter.speckleRange = 50
    config.postProcessing.temporalFilter.enable = True
    config.postProcessing.spatialFilter.enable = True
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations = 1
    config.postProcessing.thresholdFilter.minRange = depth_range[0]
    config.postProcessing.thresholdFilter.maxRange = depth_range[1]
    config.postProcessing.decimationFilter.decimationFactor = 1

    stereoDepth.initialConfig.set(config) 


    # Linking
    monoLeft.out.link(stereoDepth.left) # mono left input stereoDepth depth node
    monoRight.out.link(stereoDepth.right) # mono right input stereoDepth depth node   
    
     # Connect to device and start pipeline
    stereoDepth.depth.link(xoutDepth.input)
    camRgb.preview.link(xoutRgb.input) # color camera preview input xoutRgb node, send message device -> host

    with dai.Device(pipeline,maxUsbSpeed=dai.UsbSpeed.HIGH) as device:
        # usb2 selected, usb3 deteriorates the gps signal
        # # Set debugging level
        # device.setLogLevel(dai.LogLevel.DEBUG)
        # device.setLogOutputLevel(dai.LogLevel.DEBUG)

        
        # Output queues will be used to get the rgb frames, and depth data
        previewQueue = device.getOutputQueue("rgb", maxSize=1, blocking=False) # 'rgb' stream xoutRgb
        depthQueue = device.getOutputQueue("depth", maxSize=1, blocking=False) # depth data
        
        start_time =  int(round(time.time() * 1000))
        current_milli_time = lambda: int(round(time.time() * 1000) - start_time)        
        
        
        fps = 0
        startTime = time.monotonic()
        counter = 0
        fontType = cv2.FONT_HERSHEY_TRIPLEX

 
        while(True):
            
            inPreview = previewQueue.get()
            Frame = inPreview.getCvFrame()
            counter+=1
            current_time = time.monotonic()
            if (current_time - startTime) > 1 :
                fps = counter / (current_time - startTime)
                counter = 0
                startTime = current_time
            depth = depthQueue.get()
            depthFrame = depth.getFrame()
            
            # depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            # depthFrameColor = cv2.equalizeHist(depthFrameColor)
            # depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)            
            # cv2.imshow('depth',depthFrameColor)
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #-----------------detect the closest obstacle in screen divisions----------------------
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            """
            The depth and bgr screen has a resolution of 640x400 and is divided into # parts. 
            The depth frame coincides with the monoLeft cam with optical center pixel coordinates cu,cv. 
            The global X,Y,Z coordinates of the obstacle corresponds to the image u,v coordinates using u = fx.X/Z
            and v = fy.Y/Z. fx anf fy are the focal length along the u anf v axis in pixels. fx and fy depend on the
            image resolution of the monoLeft cam. fx=454 and fy=504 pixels in case of a 640x400 image. Given W 
            of the raster on the monoLeft image the drone will not detect the obstacle provided it is d = W.Z/(2fx)
            away from the optical axis of the monoLeft cam. In case of Z = 2m, d = 0.7m; Z = 3m, d = 1.0m

            The drone uses a FRD local frame attached to the vehicle with
            x: forward, y: right and z: down. This means that the coordinates of the closest obstacle in the # parts are 
            converted to the x,y,z of FRD using x: distance d of obstacle, y: (u-cu).d/fx and z: (v-cv).d/fy. 

            Drone-site:
            The drone proximity fence is divided in 8 sectors of 45 degrees width. The 3 monitored ROI regions are:
            front sector 0 (-22.5,22.5 degrees), sector 45 (22.5,67.5 degrees) and sector 315 (-22.5,-67.5 degrees).
            Each sector contains only 1 obstacle: the closest. ROI (214,428) covers obstacle yaw angles -13,13 degrees. ROI
            (0,214) covers the additional part of sector 0 and part of sector 45. Similar for ROI (428,640). From the obstacle
            coordinates sent the yaw angle of the obstacle is calculated which determines the sector to which the obstacle
            belongs.

            OA_MARGIN_MAX: (auto mode) min distance drone-obstacle
            AVOID_MARGIN: (gps mode-loiter) min distance drone-obstacle
            )
            """ 
            #+++++++++++++++++++++++++++++++++
            # ARRAYS: [rows, columns]
            # OPENCV: (columns,rows)
            #+++++++++++++++++++++++++++++++++
            obstacle_coordinates = np.ones((3,3))* 9999
            
            cu = 320
            cv = 200           

            # Set ROI

            for i in range(0,3):
                topLeft = [(0,5),(214,5),(428,5)] # column, row: opencv, main sector 45 degrees 
                bottomRight = [(214,350),(428,350),(640,350)]

                # print('topleft: ',topLeft,'bottomRight: ',bottomRight)
                sub_array = depthFrame[topLeft[i][1]:bottomRight[i][1],topLeft[i][0]:bottomRight[i][0]] #slice rows, colums: arrays
                obs_coord = obstac(sub_array,obstacle_coordinates,cu,cv,i,topLeft[i],depth_range)
                
                dist = round(obs_coord[0],2)
                obstacle_coordinates[i] = obs_coord[1]

                # print('iter:',i,'dist:',dist,'coordinates:',obstacle_coordinates[i])

                if flag_depth:
                    cv2.rectangle(Frame,topLeft,bottomRight,(0,0,255),3, cv2.FONT_HERSHEY_SIMPLEX)
               
                lineBot = [[(10,380),(110,380)],[(130,380),(510,380)],[(530,380),(630,380)]] # column, row
                bottomRighti = [(30,360),(300,360),(560,360)]
                                
                if chan_8 >1500:
                    if dist < dist_saf and  dist!=0.0:
                        colorText = (0,0,255)   
                    else:
                        colorText = (255,255,255)
                    cv2.putText(Frame, f"%5.2f" %dist, bottomRighti[i],  cv2.FONT_HERSHEY_TRIPLEX, 0.6, colorText)
                    cv2.line(Frame,lineBot[i][0],lineBot[i][1],colorText,5)

                    # cv2.imshow('depth',depthFrameColor)
            # obstacle_coordinates = np.round_(obstacle_coordinates,3)
            current_time_ms = current_milli_time()
            send_obstacle_distance_3D_message(conn,obstacle_coordinates,current_time_ms)
            # print(obstacle_coordinates)

            cv2.putText(Frame, "NN fps: {:.1f}".format(fps), (2, Frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))
            cv2.imshow("Frame",Frame)
            # Write the frame into the file 'output.avi'
            # out.write(Frame)
            queue2.put(Frame)
     
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                break
        # out.release()
        queue2.put(None)
        


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------thread 3------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# thread 3
# OA_TYPE 0: no Bendy Ruler, 1: Bendy Ruler active
# chan_8:show distance grid
def chan():
    global chan_8
    # chan_12_old = 1200
    param=b'OA_TYPE'# byte string of ASCI characters: O', 'A', '_', and 'T'; b'O\x00A\x00_\x00T\x00'
    read_param(conn,param)
    param=b'PRX1_TYPE'# byte string of ASCI characters: P', 'R', 'X', '1', '_', 'T', 'Y', 'P', 'E'; b'P\x00R\x00X\x001\x00_\x00T\x00Y\x00P\x00E\x00'
    # set_param(conn,param,2) # set PRX1_TYPE to 1: enable MAVLINK proximity sensor
    read_param(conn,param)
    
    while True:
        msg = read_rc_hannel(conn)
        chan_8 = msg['chan8_raw']
        # chan_12 = msg['chan12_raw']
        # if chan_12 > 1500 and chan_12 != chan_12_old:
        #     # Bendy Ruler on
        #     set_param(conn,param,1)
        #     chan_12_old = chan_12
        # if chan_12 < 1500 and chan_12 != chan_12_old:
        #     # Bendy Ruler off
        #     set_param(conn,param,0)
        #     chan_12_old = chan_12


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------thread 4------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# thread 4
def sav(queue2):
    # --- Parameters for video saving and streaming ---
    video_filename = 'oak-d.avi' # For local saving
    frame_width = 640
    frame_height = 400
    # Match this FPS to your camera/processing capabilities.
    # Your 'ai' function sets camRgb.setFps(30)
    script_fps = 20
    global flag_drone

    # --- Setup for saving to AVI file (optional, keep if you want local recording too) ---
    # file_out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), script_fps, (frame_width, frame_height))
    # if file_out.isOpened():
    #     print(f"Local recording: {video_filename} will be created.")
    # else:
    #     print(f"ERROR: Failed to open {video_filename} for local recording.")
    

    # --- Setup for GStreamer RTSP streaming to MediaMTX using rtspclientsink ---
    mediamtx_server_host = "127.0.0.1"  # MediaMTX is on the same RPi (localhost)
    mediamtx_rtsp_port = 8554           # Default RTSP port MediaMTX listens on
    mediamtx_publish_path = "stream0" # The path name you want to publish to (e.g., /test or /oakstream)
    # Bitrate for the H.264 stream (example uses 800 kbps)
    output_bitrate_kbps = 800 # For x264enc
    output_bitrate_bps = output_bitrate_kbps * 1000 # For v4l2h264enc

    # --- Choose your encoder ---

    if flag_drone:

        # Option 1: v4l2h264enc (Hardware-accelerated on RPi, preferred)
        # Note: v4l2h264enc might not support all tune/preset options like x264enc.
        # 'extra-controls' is used for parameters like bitrate.
        encoder_pipeline = (
        f"v4l2h264enc extra-controls=\"controls,video_bitrate={output_bitrate_bps}\" "
        f"! video/x-h264,profile=baseline,stream-format=byte-stream" # Ensure byte-stream for compatibility
    )
        
    else:    
        # Option 2: x264enc (Software encoder, like the gst-launch example, more CPU intensive)
            encoder_pipeline = (
                f"x264enc speed-preset=veryfast tune=zerolatency bitrate={output_bitrate_kbps}"
            )

    # GStreamer pipeline using rtspclientsink
    # This pipeline will take frames from appsrc, convert, (optionally scale), encode, and send via RTSP.
    # GStreamer pipeline for H.264 encoding and RTSP streaming
    # Ensure v4l2h264enc is available and working on your RPi4.
    # If not, try omxh264enc or, as a last resort, x264enc (software, CPU intensive).
    gst_pipeline_str = (
        f"appsrc is-live=true block=true "
        f"! video/x-raw,format=BGR,width={frame_width},height={frame_height},framerate={script_fps}/1  "
        f"! videoconvert "  # Converts BGR to a format suitable for the encoder (e.g., I420/NV12)
        f"! queue leaky=downstream max-size-buffers=2 max-size-time=0 max-size-bytes=0 "
        f"! {encoder_pipeline} "
        f"! rtspclientsink location=rtsp://{mediamtx_server_host}:{mediamtx_rtsp_port}/{mediamtx_publish_path}"
    )
    # gst_pipeline_str = ("appsrc ! videoconvert ! videoscale ! video/x-raw,width=640,height=400 "
    #                     "! x264enc speed-preset=veryfast tune=zerolatency bitrate=800  "
    #                     "! rtspclientsink location=rtsp://localhost:8554/oakstream"
    #                     )

    stream_out = cv2.VideoWriter(gst_pipeline_str, cv2.CAP_GSTREAMER, 0, script_fps, (frame_width, frame_height), True)

    if not stream_out.isOpened():
        print("ERROR: Failed to open GStreamer VideoWriter for RTSP streaming using rtspclientsink.")
        print(f"Pipeline: {gst_pipeline_str}")
        print("Please check your GStreamer installation, plugins (rtspclientsink, chosen H.264 encoder), and pipeline string.")
    else:
        print(f"Attempting to stream via RTSP PUBLISH to rtsp://{mediamtx_server_host}:{mediamtx_rtsp_port}/{mediamtx_publish_path}")
        print("INFO: On your H12 Pro, connect to: rtsp://<YOUR_RPI_IP>:8554/stream0")
    item = None
    while True:
        item = queue2.get()  # Get frame from the AI processing thread
        if item is None:    # Sentinel value to stop the thread
            print("Stopping saving/streaming thread.")
            break

        # Save to local AVI file (if opened)
        # if file_out.isOpened():
        #     file_out.write(item)

        # Stream over RTSP (if opened)
        if stream_out.isOpened():
            stream_out.write(item)
        # else:
            # Optional: Add a small sleep if streaming fails to prevent tight loop if queue is also empty
            # time.sleep(0.01)


    # Release resources
    # if file_out.isOpened():
    #     file_out.release()
    #     print(f"{video_filename} saved.")
    if stream_out.isOpened():
        stream_out.release()
        print("RTSP Streaming stopped.")







#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# --------------------------main------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

progress("INFO: Starting Vehicle communications")

import os
os.environ["MAVLINK20"] = "1"
last_obstacle_distance_sent_ms = 0

flag_drone = False #False: SITL, True: Drone

if flag_drone:
    IP = 'udpin:127.0.0.1:15550'
else:
    IP = 'udpin:192.168.1.32:15550'

conn = mavutil.mavlink_connection(
    device=IP,
    autoreconnect = True,
    source_system = 1,
    source_component = 93,
    force_connected=True,
   )

wait_conn() # send a ping to start connection @ udpin starts sending info
progress("INFO: vehicle connected")

# set_mode(conn,"LOITER")
print('verify OA_TYPE is set to 1, Bendy Ruler active')
param=b'PRX1_TYPE'# byte string of ASCI characters
set_param(conn,param,2) # set PRX1_TYPE to 1: enable MAVLINK proximity sensor
read_param(conn,param) # read param to check if set correctly

# start threads
queue2 = queue.Queue()
event = threading.Event()
t1 = threading.Thread(target=ai,args=(queue2,))
t3 = threading.Thread(target=chan)
t4 = threading.Thread(target=sav,args=(queue2,))
t3.daemon=True # has to run in console
t4.daemon=True # has to run in console, deamon threads when main thread ends
t1.start()
t3.start()
t4.start()
t1.join() # ends when task finished



