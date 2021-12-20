#!/usr/bin/env python3

import threading
import cv2
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
    0b111111000111, # type_mask (only speeds enabled), x:LSB & yaw_rate: MSB
    0, 0, 0, # x, y, z positions (not used)
    velocity_x, velocity_y, velocity_z, # x, y, z velocity in m/s
    0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
    0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)

    return msg

def progress(string):
    print(string, file=sys.stdout)
    sys.stdout.flush()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# -----------------------thread-1-----------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def ai():

    nnBlobPath = 'models/mobilenet-ssd_openvino_2021.2_6shave.blob'
    labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.createColorCamera()
    spatialDetectionNetwork = pipeline.createMobileNetSpatialDetectionNetwork()
    monoLeft = pipeline.createMonoCamera()
    monoRight = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()
    manip = pipeline.createImageManip()


    xoutRgb = pipeline.createXLinkOut()
    xoutNN = pipeline.createXLinkOut()
    xoutDepth = pipeline.createXLinkOut()

    xoutRgb.setStreamName("rgb")
    xoutNN.setStreamName("detections")
    xoutDepth.setStreamName("depth")



    # Properties
    camRgb.setPreviewSize(640, 400)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.initialControl.setManualFocus(115) # value from 0 - 255
    camRgb.initialControl.setAutoFocusMode(dai.RawCameraControl.AutoFocusMode.OFF)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    manip.initialConfig.setResize(300, 300)
    manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p) # The NN model expects BGR input

    # Setting node configs
    stereo.initialConfig.setConfidenceThreshold(255)

    spatialDetectionNetwork.setBlobPath(nnBlobPath)
    spatialDetectionNetwork.setConfidenceThreshold(0.5)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)




    # Linking
    camRgb.preview.link(manip.inputImage) # color camera preview input image mainip node
    manip.out.link(spatialDetectionNetwork.input) # manip out input spatialDetectionNetwork
    monoLeft.out.link(stereo.left) # mono left input stereo depth node
    monoRight.out.link(stereo.right) # mono right input stereo depth node

    camRgb.preview.link(xoutRgb.input) # color camera preview input xoutRgb node, send message device -> host
    spatialDetectionNetwork.out.link(xoutNN.input) # spatial detections input xoutNN node, send message device -> host
    stereo.depth.link(spatialDetectionNetwork.inputDepth) # stereo depth input spatial detection network
    spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)# stereo depth input xoutDepth node

    # Connect to device and start pipeline

    with dai.Device(pipeline) as device:

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        queue_size = 4
        previewQueue = device.getOutputQueue(name="rgb", maxSize=queue_size, blocking=False) # 'rgb' stream xoutRgb
        detectionNNQueue = device.getOutputQueue(name="detections", maxSize=queue_size, blocking=False) # 'detections stream xoutNN
        depthQueue = device.getOutputQueue(name="depth", maxSize=queue_size, blocking=False) # 'depth' stream xoutDepth
        
        startTime = time.monotonic()
        counter = 0
        fps = 0
        out = cv2.VideoWriter('oak-d.avi',cv2.VideoWriter_fourcc(*'XVID'), 20, (640,400))
        state = 'fly'
        lab = 'forward'
        fly = 400
        global velocity_y
        velocity_y=0
        fly_1=400
        k=0.66  # k=(tau/T)/(1+tau/T) tau time constant LPF, T period
                # k=0 # no filtering    
        dist_thres_cm = 240 # cm
        

        while True:
            inPreview = previewQueue.get()
            inDet = detectionNNQueue.get()
            depth = depthQueue.get()
            
            counter+=1
            current_time = time.monotonic()
            if (current_time - startTime) > 1 :
                fps = counter / (current_time - startTime)
                counter = 0
                startTime = current_time

            Frame = inPreview.getCvFrame()
            

            detections = inDet.detections

            # If the frame is available, draw bounding boxes on it and show the frame
            height = 300
            width = 300
            Z = 0
            X = 0
            Y = 0

            for detection in detections:
                # Denormalize bounding box
                x1 = int(detection.xmin * width)
                x2 = int(detection.xmax * width)
                y1 = int(detection.ymin * height)
                y2 = int(detection.ymax * height)

                # coordinates transformation: map 300x300 NN image -> preview image 400x400 + 120 (x_offset)
                X1 = int(x1*1.333+120)
                X2 = int(x2*1.333+120)
                Y1 = int(y1*1.333)
                Y2 = int(y2*1.333)
                
                
                cv2.rectangle(Frame, (X1, Y1), (X2, Y2), (255, 0, 0), cv2.FONT_HERSHEY_SIMPLEX)
                
    
                try:
                    label = labelMap[detection.label]
                except:
                    label = detection.label
                cv2.putText(Frame, str(label), (X1 + 10, Y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(Frame, "{:.2f}".format(detection.confidence*100), (X1 + 10, Y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(Frame, f"X: {int(detection.spatialCoordinates.x)} mm", (X1 + 10, Y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(Frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (X1 + 10, Y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(Frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (X1 + 10, Y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                if int(detection.spatialCoordinates.z) < Z or Z==0:
                    Z = int(detection.spatialCoordinates.z)
                    Y = int(detection.spatialCoordinates.y)
                    X = int(detection.spatialCoordinates.x)
            cv2.putText(Frame, "NN fps: {:.2f}".format(fps), (2, Frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))
            cv2.rectangle(Frame, (120, 0), (520, 400), (0, 255, 0),3, cv2.FONT_HERSHEY_SIMPLEX)

            # print('X,Y,Z:',X,Y,Z)

#===========================================================================
            # classify the input image

            if Z != 0:
                fly = Z/10 # distance in cm
            else:
                fly = 400
            fly_f = k*fly_1 + (1-k)*fly
            fly_1 = fly_f
            proba=int(fly_f)

            if state == 'avoid':
                if fly_f >= dist_thres_cm+10:
                    state='fly'
                    print(state,int(fly_f),file=f)
                    lab = 'forward'
                
            else:
                label='forward'
                # proba=fly_f
                if fly_f <= dist_thres_cm:
                    # proba=int(fly_f)
                    if X <= 0:
                        lab = 'left'
                    else:
                        lab = 'right'
                    print(lab,int(fly_f),file=f)
                    state='avoid'

            label_1 = "{} {}".format(state,lab)
            label_2 = "{} {}".format('Z(cm):',proba)
            # draw the label on the image
            cv2.putText(Frame, label_1, (10, 45),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
            cv2.putText(Frame, label_2, (10, 65),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)            
                    
            # Write the frame into the file 'output.avi'
            out.write(Frame)


            if vehicle.channels['8'] > 1400:
                if state == "fly":
                    event.clear()
    
                if state == "avoid":
                    event.set()

                    if lab == 'left':
                        velocity_y = -0.8
                    if lab == 'right':
                        velocity_y = 0.8
                
            if vehicle.channels['8'] < 1400:
                event.clear()


            cv2.imshow('Frame',Frame)

            if cv2.waitKey(1) == 27:
                break
        f.close()
        out.release()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------thread 2------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# thread 2

def slide():
    velocity_x = 0.0 # m/s
    velocity_z = 0.0
    duration = 5 # s
    while True:
        event.wait()

        if (vehicle.mode.name=='LOITER' or vehicle.mode.name=='AUTO') and vehicle.armed==True:
        # if (vehicle.mode.name=='LOITER' or vehicle.mode.name=='AUTO'):    
            flight_mode=vehicle.mode.name
            print(flight_mode,file=f)

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
f=open('/home/pi/drone_exe/drone/log_oak.txt','w')
event = threading.Event()
t1 = threading.Thread(target=ai)
t2 = threading.Thread(target=slide)
t2.daemon=True # has to run in console
t1.start()
t2.start()
t1.join()
