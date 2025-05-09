##!/usr/bin/env python3

import threading
import cv2
from simple_pid import PID
import depthai as dai
import numpy as np
import time
import sys
from pymavlink import mavutil
from dronekit import connect, VehicleMode
import queue

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ---------------------functions------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def velyaw(velocity_x, velocity_y, velocity_z,yaw):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
    0,       # time_boot_ms (not used)
    0, 0,    # target system, target component
    mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, # frame
    type_mask=mask_velyaw,
    x=0, y=0, z=0, # x, y, z positions in m 
    vx=velocity_x, vy=velocity_y, vz=velocity_z, # x, y, z velocity in m/s
    afx=0, afy=0, afz=0, # x, y, z acceleration in m/s^2
    yaw=yaw, yaw_rate=0)   # yaw, yaw_rate in rad, rad/s
    return msg

def progress(string):
    print(string, file=sys.stdout)
    sys.stdout.flush()


def traceDrone(cx,cy,cyt):
    # Kp too large -> oscillations 
    # mes_T = 1 ->  Kp=0.15, Ki=0.5 (-0.15,0.15)
    #           ->  Kp=2, Ki=1 (-0.7,0.7) 
    # mes_T = 0.4 ->  Kp=0.05, Ki=0.1 (-0.07,0.07)
    #             ->  Kp=1, Ki=1 (-0.5,0.5)

    # EK3_RNG_USE_HGT is set to 70% meaning the vertical rangefinder is used for determining the vertical height
    # up to 70% of the max height of the TF02 pro sensor (0.7*13m) provided the vxy is limited to 2 m/sec. If
    # conditions are not met the baro is used for detemining the height.
    
    pid_0 = PID(Kp=0.15,Ki=0.5,setpoint=320/100,output_limits=(-0.15,0.15))    # yaw; limit to 4 degrees
    pid_1 = PID(Kp=2,Ki=1,setpoint=cyt/100,output_limits=(-0.5,0.5))       # pitch; limit 0.5 m/sec
    
    speed_yaw = -(pid_0(cx/100))
    speed_pitch = (pid_1(cy/100))
    # speed_yaw = 0
    # speed_pitch = 0
    p, i, d = pid_0.components
    p1, i1, d1 = pid_1.components
 
    return speed_yaw, speed_pitch
	


 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# -----------------------thread-1-----------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def ai(queue):

    nnBlobPath = '/home/pi/drone_exe/drone/models//yolov8n_voc_openvino_2022.1_4shave.blob'
 
    labelMap = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.createColorCamera()
    detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
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
    camRgb.setFps(30)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    manip.initialConfig.setResize(416, 416) # size needed by mobileNet
    manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p) # The NN model expects BGR input

    # Setting node configs
    stereoDepth.initialConfig.setConfidenceThreshold(255)
    stereoDepth.setSubpixel(True)

    detectionNetwork.setBlobPath(nnBlobPath)
    detectionNetwork.setConfidenceThreshold(0.5)
    detectionNetwork.input.setBlocking(False)
    detectionNetwork.setIouThreshold(0.5)
    detectionNetwork.setNumClasses(20) #voc=20, coco=80
    detectionNetwork.setCoordinateSize(4)
    detectionNetwork.setAnchors([])
    detectionNetwork.setAnchorMasks({})
    detectionNetwork.setNumInferenceThreads(2)
   

    objectTracker.setDetectionLabelsToTrack([14])  # track #0: only person coco; #14:person voc
    # objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    objectTracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)
    objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID) # smallest available ID


    # Linking
    camRgb.preview.link(manip.inputImage) # color camera preview input image mainip node
    manip.out.link(detectionNetwork.input) # manip out input detectionNetwork
    monoLeft.out.link(stereoDepth.left) # mono left input stereoDepth depth node
    monoRight.out.link(stereoDepth.right) # mono right input stereoDepth depth node
    detectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)
    detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
    detectionNetwork.out.link(objectTracker.inputDetections)
    
    
    
    # Connect to device and start pipeline
    stereoDepth.depth.link(xoutDepth.input)
    objectTracker.out.link(trackerOut.input) # objectTracker.out input trackerOut, send message device -> host 
    camRgb.preview.link(xoutRgb.input) # color camera preview input xoutRgb node, send message device -> host
    

      

    with dai.Device(pipeline,maxUsbSpeed=dai.UsbSpeed.HIGH) as device:
        # usb2 selected, usb3 deteriorates the gps signal
        # # Set debugging level
        # device.setLogLevel(dai.LogLevel.DEBUG)
        # device.setLogOutputLevel(dai.LogLevel.DEBUG)

        # Output queues will be used to get the rgb frames, tracklet and depth data
        previewQueue = device.getOutputQueue("rgb", maxSize=1, blocking=False) # 'rgb' stream xoutRgb
        tracklets = device.getOutputQueue("tracklets", maxSize=1, blocking=False) # object & track data
        depthQueue = device.getOutputQueue("depth", maxSize=1, blocking=False) # depth data
        #+++++++++++++++++++++++++++
        # init variables
        startTime = time.monotonic()
        counter = 0
        fps = 0
        color = (0,0,0)
        global state,flag_track,cx,hgn,hg,hgt,altCur,altTar,cy,cyt
        cx = 320 # pixels
        cy = 200
        ID = 0
        # hgn = 1 # pixels
        # hg = 100
        # hgt = 100
        cyt = 200
        altCur = 3 # m
        altTar = 3
        flag_track = False
        flagOnce = True
        state = 'fly'
        lab = 'forward'
        y2 = 200
        k=0.66  # k=(tau/T)/(1+tau/T) tau time constant LPF, T period; k=0 # no filtering 
        per_center_1 = 10 
        #+++++++++++++++++++++++++++
        # dist_thres_mm = 3000 # mm
        dist_thres_mm = 100 # mm
        dist_safe = 200 # mm
        #+++++++++++++++++++++++++++
        flag = False
        flagCent = False
        crit = 10 # threshold number of pixels detected
 
        while(True):
            
            inPreview = previewQueue.get()
            # print('inPreview:', inPreview)
            Frame = inPreview.getCvFrame()
            track = tracklets.get()
            depth = depthQueue.get()
            depthFrame = depth.getFrame()
            
            # depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            # depthFrameColor = cv2.equalizeHist(depthFrameColor)
            # depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)            
            # print('frame',Frame,depthFrame)

            # 2 arrays derived from depthFrame with distances smaller than dist_thres & dist_thres+200 mmm
            dist = cv2.inRange(depthFrame,200,dist_thres_mm) # detect 200<dist(mm)<dist_thres_mm
            dist_fl = cv2.inRange(depthFrame,200,dist_thres_mm+dist_safe) # needed for defining a hysteresis 200mm

            # rectangle window (center_win) defined at 3m from drone: width 2m height 1m
            # 2 rectangles (left_win & right_win) 0.66m*1m left and right form center_win
            # center of center_win rectangle (307,235) determined by homography from center of inPreview (320,200)
            left_win = dist[56:370,73:166] # slice: rows,columns
            right_win = dist[56:370,469:557]
            center_win =dist[56:370,166:469]
            center_win_fl = dist_fl[56:370,166:469]

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
                # roi = t.roi.denormalize(400, 400)
                roi = t.roi.denormalize(416, 416)
                x1 = int(roi.topLeft().x)
                y1 = int(roi.topLeft().y)
                x2 = int(roi.bottomRight().x)
                y2 = int(roi.bottomRight().y)

                try:
                    label = labelMap[t.label]
                except:
                    label = t.label
 
                # coordinates transformation: map 416x416 NN image -> preview image 400x400 + 120 (x_offset)
                x1 = int(x1*0.96+120)
                x2 = int(x2*0.96+120)
                y1 = int(y1*0.96)
                y2 = int(y2*0.96)
            
            
                cv2.putText(Frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(Frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(Frame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.rectangle(Frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                if t.status.name == "TRACKED" and t.id==ID:
                # if t.status.name == "TRACKED": 
                    cx = (x2-x1)//2+x1
                    cy = (y2-y1)//2+y1
                    # hg = (y2-y1)
                    # hgn = hg/hgt
                    altCur = vehicle.location.global_relative_frame.alt
                    flag_track = True 
                    lab = "track"
                    cv2.rectangle(Frame, (x1, y1), (x2, y2), (0,0,255),2, cv2.FONT_HERSHEY_SIMPLEX)

                else:
                    flag_track = False
                    lab = "no track"
                # cv2.circle(Frame, (cx, cy), 5, (0, 0, 255), -1)
                # cv2.rectangle(Frame, (220,220), (420,320),(255,0,255),2,cv2.FONT_HERSHEY_SIMPLEX)  #track window for cx,cy           
                # cv2.putText(Frame, f"cx: {int(cx)}", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                # cv2.putText(Frame, f"hg: {int(hg)}", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                # cv2.putText(Frame, f"alt: {int(altCur*100)}", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            
            cv2.putText(Frame, "NN fps: {:.2f}".format(fps), (2, Frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,0,0))
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

            if state == 'stop':
                if not(flagCent or y2>395):# lower coordinate detection rectangle not below detection window
                    state = 'fly'
                    # lab = 'track'

            if state == 'fly':
                # lab = 'track'
                if flagCent or y2>395:
                    state = 'stop'
                    lab = 'no track'


            if vehicle.channels['8'] > 1400:
                colo = (0,0,255)
                if flagOnce and flag_track:
                    # ctx = cx
                    cyt = cy
                    flagOnce = False
                    flag = True
                # if state == "fly":
                    event.set()
                
                
            if vehicle.channels['8'] < 1400:
                event.clear()
                flag = False
                flagOnce = True
                colo = (0,0,0)
                
                while vehicle.mode.name=='GUIDED':
                    vehicle.mode=VehicleMode("LOITER")
                    print('Waiting for mode LOITER',file=f)
                    time.sleep(0.5)                
            

            if flag:
                txt = "on"
                # colo = (0,0,255)
                # traceDrone(cx,cy)
            else:
                txt = "off"
                # colo = (0,255,0)
                
            label_1 = "{} {}".format(state,lab)
            label_3 = "{} {}".format('track:',txt)
            # colo = (0,255,0)
            # draw the label on the image
            cv2.putText(Frame, label_1, (10, 45),  cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0))
            cv2.putText(Frame, label_3, (10, 65),  cv2.FONT_HERSHEY_TRIPLEX, 0.6, colo)
            cv2.putText(Frame, str(per_left), (50, 360),  cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 255, 0))
            cv2.putText(Frame, str(per_center), (320, 360),  cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 255, 0))
            cv2.putText(Frame, str(per_right), (580, 360),  cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 255, 0))

            # Write the frame into the file 'output.avi'
            # out.write(Frame)
            queue.put(Frame)

            cv2.imshow('Frame',Frame)
            # cv2.imshow('center_win',center_win)
            if cv2.waitKey(1) == 27:
                break # interrupt loop by pressing 'esc' key
        f.close()
        queue.put(None)
        # out.release()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------thread 2------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# thread 2

def slide():
    posZ = 0
    mes_T = 1 # periode messages are sent

    while True:
        event.wait()

        #if (vehicle.mode.name=='LOITER' or vehicle.mode.name=='AUTO') and vehicle.armed==True:
        if (vehicle.mode.name=='LOITER'):    
            flight_mode=vehicle.mode.name
            print(flight_mode,file=f)
            # print(flight_mode)
            
            vehicle.mode=VehicleMode("GUIDED")
            while not vehicle.mode.name=='GUIDED':
                print('Waiting for mode GUIDED',file=f)
                time.sleep(0.5)
            print(vehicle.mode.name,file=f)
            msg=velyaw(0,0,0,0)
            vehicle.send_mavlink(msg) 
            count=0

            while event.is_set():

                if state == "fly":
                    if flag_track:
                        print('cx,cy:',cx,cy,file=f) # log
                        speed_yaw,speed_pitch = traceDrone(cx,cy,cyt)
                        print('speed_yaw,speed_pitch:',speed_yaw,speed_pitch,file=f)
                        msg=velyaw(speed_pitch,0,0,speed_yaw)
                        vehicle.send_mavlink(msg)
                        time.sleep(mes_T) # message frequency not too high                      

                    else:
                        msg=velyaw(0,0,0,0)
                        print('No Track,altCur:',altCur,file=f)
                        vehicle.send_mavlink(msg)
                        time.sleep(mes_T) # message frequency not too high

                if state == "stop":
                        msg=velyaw(0,0,0,0)
                        print('STOP, altCur:',altCur,file=f)
                        vehicle.send_mavlink(msg)
                        time.sleep(mes_T) # message frequency not too high

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------thread 3------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# thread 3
def iden():
    global ID
    ID = 0
    while True:
        # ID = 100
        event.wait()
        # print('ID:',ID)
        while event.is_set():
            # input the object to track
            
            ID_1 = input('enter id number:')
            ID = int(ID_1)
            print('ID:',ID)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------thread 4------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# thread 4
def sav(queue):
    out = cv2.VideoWriter('oak-d.avi',cv2.VideoWriter_fourcc(*'XVID'), 20, (640,400))
    print("oak-d.avi created")
    item = None
    while True:    
        item = queue.get()
        if item is None:
            out.release()
            break
        out.write(item)
        
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
VY_IGNORE = 32
AX_IGNORE = 64
AY_IGNORE = 128
AZ_IGNORE = 256
FORCE_SET = 512
YAW_IGNORE = 1024 
YAW_RATE_IGNORE =2048

mask_velyaw = X_IGNORE | Y_IGNORE | Z_IGNORE | AX_IGNORE | AY_IGNORE | AZ_IGNORE | YAW_RATE_IGNORE # use velocity and Yaw
print('type_mask_velyaw=',mask_velyaw) # 2503

progress("INFO: Starting Vehicle communications")
# connect vehicle to access point
##vehicle = connect('udpin:192.168.1.61:14550', wait_ready=False)
vehicle = connect('udpin:127.0.0.1:15550', wait_ready=False,source_system=1,source_component=10) # on rpi
# vehicle = connect('udpin:127.0.0.1:14550', wait_ready=False)
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
print(" Altitude: %s" % vehicle.location.global_relative_frame.alt)
print(" Velocity: %s" % vehicle.velocity)
print(" GPS: %s" % vehicle.gps_0)
print(" Flight mode currently: %s" % vehicle.mode.name)

# start threads
f=open('oak_log.txt','w')
queue = queue.Queue()
event = threading.Event()
t1 = threading.Thread(target=ai,args=(queue,))
t2 = threading.Thread(target=slide)
t3 = threading.Thread(target=iden)
t4 = threading.Thread(target=sav,args=(queue,))
t2.daemon=True # has to run in console
t3.daemon=True # has to run in console
t4.daemon=True # has to run in console, deamon threads when main thread ends
t1.start()
t2.start()
t3.start()
t4.start()
t1.join() # ends when task finished



