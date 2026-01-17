import cv2
import threading
import numpy as np
import depthai as dai
import collections
import time
collections.MutableMapping = collections.abc.MutableMapping
from dronekit import connect, VehicleMode
from pymavlink import mavutil
from numba import njit
import queue
import sys

class ObstacleAvoider:
    def __init__(self, width, height, obstacle_threshold, flow_scale, smoothing_factor=0.1, depth_range=(200,10000),dist_threshold = 2):
        self.width = width
        self.height = height
        self.obstacle_threshold = obstacle_threshold
        self.flow_scale = flow_scale
        self.old_gray = None
        self.mask = None
        self.smoothed_linear_velocity = 0
        self.smoothed_angular_velocity = 0
        self.smoothing_factor = smoothing_factor
        self.depth_range = depth_range # mm
        self.dist_threshold = dist_threshold
        


    def initialize(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.old_gray = gray
        self.mask = np.zeros_like(frame)

    def get_flow_field(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(self.old_gray, gray, None, 0.5, 2, 10, 3, 5, 1.1, 0)
        self.old_gray = gray
        return flow

    def disp(self,frame,flow):
        # Create mask
        hsv_mask = np.zeros_like(frame)
        # Make image saturation to a maximum value
        hsv_mask[..., 1] = 255
        # Compute magnite and angle of 2D vector
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])    
        # Set image hue value according to the angle of optical flow
        hsv_mask[..., 0] = ang * 180 / np.pi / 2
        # Set value as per the normalized magnitude of optical flow
        hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # Convert to rgb
        rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
        return rgb_representation
        


    def check_for_obstacles(self, flow, obst_coord):
        obstacle_detected = False
        direction = "forward"  # Default
        avg_flow_x = np.mean(flow[..., 0])
        avg_flow_y = np.mean(flow[..., 1])
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_magnitude = np.mean(magnitude)
        
        left_flow = flow[:, :self.width // 3]
        center_flow = flow[:, self.width // 3: 2 * self.width // 3]
        right_flow = flow[:, 2 * self.width // 3:]

        avg_magnitude_left = np.mean(cv2.cartToPolar(left_flow[..., 0], left_flow[..., 1])[0]) if left_flow.size > 0 else 0
        avg_magnitude_center = np.mean(cv2.cartToPolar(center_flow[..., 0], center_flow[..., 1])[0]) if center_flow.size > 0 else 0
        avg_magnitude_right = np.mean(cv2.cartToPolar(right_flow[..., 0], right_flow[..., 1])[0]) if right_flow.size > 0 else 0

        # Obstacle detection based on depth data
        depth_obstacle = False
        depth_direction = "forward"

        # Obstacle Coordinates: obstacle_coordinates[i][0] = d, x = (u-cu).d/fx, y = (v-cv).d/fy
        """
        The detection of 'depth_obstacle' using stereo cams is checked first to determine the direction.
        Next optical flow obstacle is used to determine the direction. If one wants optical flow detects
        obstacles at greater distances, one has to use smaller 'obstacle threshold' parameters and small
        'dist_threshold' values

        """
        dist_left = np.sqrt((obst_coord[0][0])**2+(obst_coord[0][1])**2+(obst_coord[0][2])**2)
        dist_center = np.sqrt((obst_coord[1][0])**2+(obst_coord[1][1])**2+(obst_coord[1][2])**2)
        dist_right = np.sqrt((obst_coord[2][0])**2+(obst_coord[2][1])**2+(obst_coord[2][2])**2)
        
        if dist_center < self.dist_threshold and self.depth_range[0]/1000 <= dist_center <= self.depth_range[1]/1000: 
            depth_obstacle = True
            if dist_left < dist_right and dist_left < dist_center:
                depth_direction = "right"
            elif dist_right < dist_left and dist_right < dist_center:
                depth_direction = "left"
            else:
                depth_direction = "back" # or stop.
        
        

        if (avg_magnitude > self.obstacle_threshold) or depth_obstacle:
            obstacle_detected = True
            if depth_obstacle:
                direction = depth_direction # set direction using closest object based on depth
            elif avg_magnitude_left > avg_magnitude_right and avg_magnitude_left > avg_magnitude_center:
                direction = "right"
            elif avg_magnitude_right > avg_magnitude_left and avg_magnitude_right > avg_magnitude_center:
                direction = "left"
            else:
                direction = "back" # or stop.  Could also use flow direction
        return obstacle_detected, direction, avg_flow_x, avg_flow_y, avg_magnitude

    def get_motion_commands(self, avg_flow_x, avg_flow_y):
        linear_velocity = 0.5
        angular_velocity = -avg_flow_x * self.flow_scale

        # Apply smoothing
        self.smoothed_linear_velocity = (self.smoothing_factor * linear_velocity) + (1 - self.smoothing_factor) * self.smoothed_linear_velocity
        self.smoothed_angular_velocity = (self.smoothing_factor * angular_velocity) + (1 - self.smoothing_factor) * self.smoothed_angular_velocity

        return self.smoothed_linear_velocity, self.smoothed_angular_velocity

        
    def process_frame(self, frame, vehicle, obst_coord):
        if self.old_gray is None:
            self.initialize(frame)
            return False

        flow = self.get_flow_field(frame)
        obstacle_detected, direction, avg_flow_x, avg_flow_y, avg_magnitude = self.check_for_obstacles(flow,obst_coord)
        rgb_representation = self.disp(frame,flow)
        cv2.imshow("rgb_representation", rgb_representation)
        

        if obstacle_detected:
            if direction == "right":
                linear_velocity = 0.2
                angular_velocity = 0.3 # Turn right
            elif direction == "left":
                linear_velocity = 0.2
                angular_velocity = -0.3  # Turn left
            else:
                linear_velocity = -0.2
                angular_velocity = 0 # Stop or back up

            set_velocity(vehicle, linear_velocity, 0, angular_velocity)  # Fly "forward" at specified velocity and yaw
            print(f"Linear Velocity: {linear_velocity:.2f}, Angular Velocity: {angular_velocity:.2f}")
            return obstacle_detected
        else:
            set_velocity(vehicle, 0.5, 0, 0)
            return obstacle_detected


def set_velocity(vehicle, vx, vy, yaw_rate):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,   # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_NED, # frame
        type_mask=mask_velocity, # type_mask (only speeds enabled, but now also yaw_rate)
        x=0, y=0, z=0, # x, y, z positions in m 
        vx=vx, vy=vy, vz=0, # x, y, z velocity in m/s
        afx=0, afy=0, afz=0, # x, y, z acceleration in m/s^2
        yaw=0, yaw_rate=yaw_rate)   # yaw, yaw_rate in rad, rad/s
    vehicle.send_mavlink(msg)
    vehicle.flush()

def progress(string):
    print(string, file=sys.stdout)
    sys.stdout.flush()

@njit
def obstac(arr,obs_coord,cu,cv,row,topLeft,depth_range):
    h = arr.shape[0] # rows
    w = arr.shape[1] # columns
    for j in range(0,h,50):      
        for k in range(0,w,50):
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

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#------------------------thread 1------------------------------------------------------------------------------------------
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def cam(queue1):
    # camera_width = 640
    # camera_height = 400
    camera_width = 160
    camera_height = 100    
    obstacle_threshold = 1  # Adjust based on tests, larger values -> object closer
    flow_scale = 0.1 # Adjust based on tests
    smoothing_factor=0.1
    depth_range=(200,10000)
    dist_threshold = 0.6 #in m
    fps = 0
    startTime = time.monotonic()
    counter = 0
    flagOnce = True

    # Initialize ObstacleAvoider
    avoider = ObstacleAvoider(camera_width, camera_height, obstacle_threshold, flow_scale,smoothing_factor, depth_range,dist_threshold)


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

    # Setting node configs
    stereoDepth.initialConfig.setConfidenceThreshold(130)
    stereoDepth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereoDepth.setSubpixel(True)
    stereoDepth.setLeftRightCheck(True)

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

        while True:
            # ret, frame = cap.read()
            # if not ret:
            #     print("Error: Could not read frame")
            #     break
            counter+=1
            current_time = time.monotonic()
            if (current_time - startTime) > 1 :
                fps = counter / (current_time - startTime)
                counter = 0
                startTime = current_time
            
            inPreview = previewQueue.get()
            frame = inPreview.getCvFrame()
            frame_display = frame.copy()
            frame_display = cv2.resize(frame_display, (camera_width,camera_height))

            depth = depthQueue.get()
            depthFrame = depth.getFrame()
            #+++++++++++++++++++++++++++++++++
            # ARRAYS: [rows, columns]
            # OPENCV: (columns,rows)
            #+++++++++++++++++++++++++++++++++
            obst_coord = np.ones((3,3))* 9999
            colorText = (255,255,255)
           
            cu = 320
            cv = 200

            for i in range(0,3):
                topLeft = [(0,5),(214,5),(428,5)] # column, row: opencv, main sector 45 degrees 
                bottomRight = [(214,350),(428,350),(640,350)]

                # print('topleft: ',topLeft,'bottomRight: ',bottomRight)
                sub_array = depthFrame[topLeft[i][1]:bottomRight[i][1],topLeft[i][0]:bottomRight[i][0]] #slice rows, colums: arrays
                obs_coord = obstac(sub_array,obst_coord,cu,cv,i,topLeft[i],depth_range)
                
                dist = round(obs_coord[0],2)
                obst_coord[i] = obs_coord[1]

                # print('iter:',i,'dist:',dist)
 
                lineBot = [[(10,380),(110,380)],[(130,380),(510,380)],[(530,380),(630,380)]] # column, row
                bottomRighti = [(30,360),(300,360),(560,360)]
                                

                if dist < dist_threshold and  dist!=0.0:
                    colorText = (0,0,255)   
                else:
                    colorText = (255,255,255)
                cv2.putText(frame, f"%5.2f" %dist, bottomRighti[i],  cv2.FONT_HERSHEY_TRIPLEX, 0.6, colorText)
                cv2.line(frame,lineBot[i][0],lineBot[i][1],colorText,5)            

            obstacle_detected = avoider.process_frame(frame_display, vehicle, obst_coord)
            
            
            # Display
            if obstacle_detected:
                text = "Obstacle Detected! Steering..."
            else:
                text = "Cruising..."

            if flag_drone:
                if vehicle.channels['8'] > 1400:
                    if flagOnce:
                        flagOnce = False
                    if (vehicle.mode.name=='LOITER'):    
                        flight_mode=vehicle.mode.name
                        vehicle.mode=VehicleMode("GUIDED")
                        while not vehicle.mode.name=='GUIDED':
                            time.sleep(0.5)
                if vehicle.channels['8'] < 1400:
                    flagOnce = True
                    while vehicle.mode.name=='GUIDED':
                        vehicle.mode=VehicleMode("LOITER")
                        time.sleep(0.5)

            cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, 390), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)    

            cv2.imshow("frame", frame)
            queue1.put(frame)

            # Exit on pressing 'esc'
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                break
        queue1.put(None)
            
            
      


                        
           
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#------------------------thread 2------------------------------------------------------------------------------------------
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def obst(queue1):
    # --- Parameters for video saving and streaming ---
    frame_width = 640
    frame_height = 400
    # Match this FPS to your camera/processing capabilities.
    # Your 'ai' function sets camRgb.setFps(30)
    script_fps = 20
    global flag_drone


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
        item = queue1.get()  # Get frame from the AI processing thread
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

      
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#------------------------main------------------------------------------------------------------------------------------
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Initialize DroneKit
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

mask_velocity = X_IGNORE | Y_IGNORE | Z_IGNORE | AX_IGNORE | AY_IGNORE | AZ_IGNORE | YAW_IGNORE # use velocity and Yaw
print('type_set_velocity=',mask_velocity) # 1479
global flag_drone
progress("INFO: VERIFY PRX1_TYPE=0")
time.sleep(5)
progress("INFO: Starting Vehicle communications")

flag_drone = False #False: SITL, True: Drone
if flag_drone:
    IP = 'udpin:127.0.0.1:15550'
else:
    IP = 'udpin:192.168.1.32:15550'
connection_string = IP  # Change this to your drone's connection
print('Connecting to vehicle on: %s' % connection_string)
vehicle = connect(connection_string, wait_ready=False)

# Get all vehicle attributes (state)
print("\nGet all vehicle attribute values:")
print(" Autopilot Firmware version: %s" % vehicle.version)
print("   Major version number: %s" % vehicle.version.major)
print("   Minor version number: %s" % vehicle.version.minor)
print("   Patch version number: %s" % vehicle.version.patch)
print("   Release type: %s" % vehicle.version.release_type())
print("   Release version: %s" % vehicle.version.release_version())
print("   Stable release?: %s" % vehicle.version.is_stable())
attitude = vehicle.attitude
print(" %s" % attitude)
print("yaw =",attitude.yaw)
print(" Altitude: %s" % vehicle.location.global_relative_frame.alt)
print(" %s" % vehicle.location.global_frame)
location = vehicle.location.local_frame
print(" %s" % location)    #NED
print("location north = ",location.north)
print(" Velocity: %s" % vehicle.velocity)
print(" GPS: %s" % vehicle.gps_0)
print(" Flight mode currently: %s" % vehicle.mode.name) 

queue1 = queue.Queue()
event = threading.Event()
t1 = threading.Thread(target=cam,args=(queue1,))
t2 = threading.Thread(target=obst,args=(queue1,))
t2.daemon=True # has to run in console

t1.start()
t2.start()
t1.join() # ends when task finished    
 



