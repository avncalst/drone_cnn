##!/usr/bin/env python3

import threading
import cv2
from cv2 import FILE_STORAGE_READ
import depthai as dai
import numpy as np
import time
import sys
from pymavlink import mavutil
from dronekit import connect, VehicleMode


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
    #print('mask:',mask)
    #print('vy:',velocity_y)
    return msg

def progress(string):
    print(string, file=sys.stdout)
    sys.stdout.flush()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# -----------------------thread-1-----------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def ai():

    nnBlobPath = '/home/pi/drone_exe/drone/models/mobilenet-ssd_openvino_2021.2_6shave.blob'
    # nnBlobPath = 'models/mobilenet-ssd_openvino_2021.2_6shave.blob'
    labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.createColorCamera()
    spatialDetectionNetwork = pipeline.createMobileNetSpatialDetectionNetwork()
    monoLeft = pipeline.createMonoCamera()
    monoRight = pipeline.createMonoCamera()
    stereoDepth = pipeline.createStereoDepth()
    manip = pipeline.createImageManip()
    objectTracker = pipeline.createObjectTracker()


    xoutRgb = pipeline.createXLinkOut()
    trackerOut = pipeline.createXLinkOut()
    xoutDepth = pipeline.createXLinkOut()
 
    xoutRgb.setStreamName("rgb")
    trackerOut.setStreamName("tracklets")
    xoutDepth.setStreamName("depth")




    # Properties
    camRgb.setPreviewSize(640, 400)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    # camRgb.initialControl.setManualFocus(115) # value from 0 - 255
    # camRgb.initialControl.setAutoFocusMode(dai.RawCameraControl.AutoFocusMode.OFF)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    manip.initialConfig.setResize(300, 300) # size needed by mobileNet
    manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p) # The NN model expects BGR input

    # Setting node configs
    stereoDepth.initialConfig.setConfidenceThreshold(255)
    stereoDepth.setSubpixel(True)

    spatialDetectionNetwork.setBlobPath(nnBlobPath)
    spatialDetectionNetwork.setConfidenceThreshold(0.5)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100) # depth in mm
    spatialDetectionNetwork.setDepthUpperThreshold(7000)
    

    objectTracker.setDetectionLabelsToTrack([15])  # track #15: only person
    objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID) # smallest available ID


    # Linking
    camRgb.preview.link(manip.inputImage) # color camera preview input image mainip node
    manip.out.link(spatialDetectionNetwork.input) # manip out input spatialDetectionNetwork
    monoLeft.out.link(stereoDepth.left) # mono left input stereoDepth depth node
    monoRight.out.link(stereoDepth.right) # mono right input stereoDepth depth node
    spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)
    spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
    spatialDetectionNetwork.out.link(objectTracker.inputDetections)
    stereoDepth.depth.link(spatialDetectionNetwork.inputDepth)
    
    
    # Connect to device and start pipeline
    objectTracker.out.link(trackerOut.input) # objectTracker.out input trackerOut, send message device -> host 
    camRgb.preview.link(xoutRgb.input) # color camera preview input xoutRgb node, send message device -> host
    spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input) # spatialDetectionnetwork.passthrough input xoutDepth,send message -> host


      

    with dai.Device(pipeline) as device:

        # Output queues will be used to get the rgb frames, tracklet and depth data
        previewQueue = device.getOutputQueue("rgb", maxSize=4, blocking=False) # 'rgb' stream xoutRgb
        tracklets = device.getOutputQueue("tracklets", 4, False) # object & track data
        depthQueue = device.getOutputQueue("depth", 4, False) # depth data
        # init variables
        startTime = time.monotonic()
        out = cv2.VideoWriter('oak-d.avi',cv2.VideoWriter_fourcc(*'XVID'), 20, (640,400))
        counter = 0
        fps = 0
        color = (255,0,0)
        state = 'fly'
        lab = 'forward'
        fly = 400
        global velocity_y
        velocity_y=0
        fly_1=400
        k=0.66  # k=(tau/T)/(1+tau/T) tau time constant LPF, T period; k=0 # no filtering 
        per_center_1 = 10 
        dist_thres_mm = 3000 # mm
        dist_safe = 200 # mm
        flag = False
        flagLeft = False
        flagRight = False
        flagCent = False
        crit = 10 # threshold number of pixels detected
         

        while True:
            inPreview = previewQueue.get()
            track = tracklets.get()
            depth = depthQueue.get()
            depthFrame = depth.getFrame()
            Frame = inPreview.getCvFrame()
            # depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            # depthFrameColor = cv2.equalizeHist(depthFrameColor)
            # depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)            

            # 2 arrays derived from depthFrame with distances smaller than dist_thres & dist_thres+200 mmm
            dist = cv2.inRange(depthFrame,200,dist_thres_mm) # detect 200<dist(mm)<dist_thres_mm
            dist_fl = cv2.inRange(depthFrame,200,dist_thres_mm+dist_safe) # needed for defining a hysteresis 200mm

            # rectangle window (center_win) defined at 3m from drone: width 2m height 1m
            # 2 rectangles (left_win & right_win) 0.66m*1m left and right form center_win
            # center of center_win rectangle (307,235) determined by homography from center of inPreview (320,200)
            left_win = dist[151:319,56:156]
            right_win = dist[151:319,458:558]
            center_win =dist[151:319,156:458]
            center_win_fl = dist_fl[151:319,156:458]

            # check whether objects are closer than dist_thres(+200 mm)
            count_left = cv2.countNonZero(left_win)
            count_right = cv2.countNonZero(right_win)
            count_center = cv2.countNonZero(center_win)
            count_center_fl = cv2.countNonZero(center_win_fl)

            # normalize
            per_left = count_left//100
            per_right = count_right//100
            per_center = count_center//100 # 200<dist(mm)<dist_thres_mm
            per_center_fl = count_center_fl//100 # 200<dist(mm)<dist_thres_mm+dist_safe

            # LPF
            per_center_f = k*per_center_1 + (1-k)*per_center
            per_center_1 = per_center_f                        

            # print('per_left,per_right,per_center',per_left,per_right,per_center, file=f)
            
            counter+=1
            current_time = time.monotonic()
            if (current_time - startTime) > 1 :
                fps = counter / (current_time - startTime)
                counter = 0
                startTime = current_time


            # If the frame is available, draw bounding boxes on it and show the frame
            trackletsData = track.tracklets
            for t in trackletsData:
                roi = t.roi.denormalize(300, 300)
                x1 = int(roi.topLeft().x)
                y1 = int(roi.topLeft().y)
                x2 = int(roi.bottomRight().x)
                y2 = int(roi.bottomRight().y)

                try:
                    label = labelMap[t.label]
                except:
                    label = t.label
 
                # coordinates transformation: map 300x300 NN image -> preview image 400x400 + 120 (x_offset)
                x1 = int(x1*1.333+120)
                x2 = int(x2*1.333+120)
                y1 = int(y1*1.333)
                y2 = int(y2*1.333)
            
                cv2.putText(Frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(Frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(Frame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.rectangle(Frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
                area = (x2-x1)*(y2-y1) # rectangle surface

                cv2.putText(Frame, f"area: {int(area)}", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(Frame, f"Z: {int(t.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

            cv2.putText(Frame, "NN fps: {:.2f}".format(fps), (2, Frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
            cv2.rectangle(Frame, (120, 0), (520, 400), (0, 255, 0),3, cv2.FONT_HERSHEY_SIMPLEX)

#===========================================================================
        #avoidance algorithm
        
            if per_left < crit:
                col = (0,255,0)
                flagLeft = False # no objects on left side closer than dist_thres
            else:
                col = (0,0,255)
                flagLeft = True # objects on left side closer than dist_thres
            cv2.line(Frame,(10,380),(110,380),col,5)

            if per_right < crit:
                col = (0,255,0)
                flagRight = False # no objects on right side closer than dist_thres
            else:
                col = (0,0,255)
                flagRight = True # objects on right side closer than dist_thres
            cv2.line(Frame,(530,380),(630,380),col,5)

            if per_center_fl < crit:
                col = (0,255,0) # green
                flagCent = False # no objects closer than dist_thres + dist_safe
            if per_center_f >= crit:
                col = (0,0,255) # red
                flagCent = True # objects closer than dist_thres
            cv2.line(Frame,(130,380),(510,380),col,5)

            if state == 'avoid':
                if not(flagCent):
                    state = 'fly'
                    lab = 'forward'

            if state == 'fly':
                lab = 'forward'
                if flagCent:
                    state = 'avoid'
                    if flagLeft and flagRight:
                        lab = 'stop'
                    if not(flagLeft) and flagRight:
                        lab = 'left'
                    if flagLeft and not(flagRight):
                        lab = 'right'
                    if not(flagLeft) and not(flagRight):
                        lab = 'right' 

            if vehicle.channels['8'] > 1400:
                flag = True
                if state == "fly":
                    event.clear()
    
                if state == "avoid":
                    event.set()
                    if lab == 'left':
                        velocity_y = -0.6
                    if lab == 'right':
                        velocity_y = 0.6
                    if lab == 'stop':
                        velocity_y = 0
                
            if vehicle.channels['8'] < 1400:
                event.clear()
                flag = False
            

            if flag:
                txt = "on"
                colo = (0,0,255)
            else:
                txt = "off"
                colo = (0,255,0)
                
            label_1 = "{} {}".format(state,lab)
            label_3 = "{} {}".format('avoid:',txt)

            # draw the label on the image
            cv2.putText(Frame, label_1, (10, 45),  cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 255, 0))
            cv2.putText(Frame, label_3, (10, 65),  cv2.FONT_HERSHEY_TRIPLEX, 0.6, colo)
            cv2.putText(Frame, str(per_left), (50, 360),  cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 255, 0))
            cv2.putText(Frame, str(per_center), (320, 360),  cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 255, 0))
            cv2.putText(Frame, str(per_right), (580, 360),  cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 255, 0))

            # Write the frame into the file 'output.avi'
            out.write(Frame)

            cv2.imshow('Frame',Frame)
            # cv2.imshow('left_win',left_win)
            # cv2.imshow('right_win',right_win)
            # cv2.imshow('center_win',center_win)
            # cv2.imshow("depth",depthFrameColor )

            if cv2.waitKey(1) == 27:
                break # interrupt loop by pressing 'esc' key
        f.close()
        out.release()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------thread 2------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# thread 2

def slide():
    velocity_x = 0.0 # m/s
    velocity_z = 0.0
    
    while True:
        event.wait()

        #if (vehicle.mode.name=='LOITER' or vehicle.mode.name=='AUTO') and vehicle.armed==True:
        if (vehicle.mode.name=='LOITER' or vehicle.mode.name=='AUTO'):    
            flight_mode=vehicle.mode.name
            print(flight_mode,file=f)
            # print(flight_mode)
            
            vehicle.mode=VehicleMode("GUIDED")
            while not vehicle.mode.name=='GUIDED':
                print('Waiting for mode GUIDED',file=f)
                time.sleep(0.5)
            print(vehicle.mode.name,file=f)
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
            # msg=slide_velocity(0,0,0)
            # vehicle.send_mavlink(msg)                
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
vehicle = connect('udpin:127.0.0.1:15550', wait_ready=False,source_system=1,source_component=10)
#vehicle = connect('udpout:127.0.0.1:15667', wait_ready=False,source_system=1,source_component=1)
vehicle.initialize(10,30)
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
f=open('oak_log.txt','w')
event = threading.Event()
t1 = threading.Thread(target=ai)
t2 = threading.Thread(target=slide)
t2.daemon=True # has to run in console
t1.start()
t2.start()
t1.join()
