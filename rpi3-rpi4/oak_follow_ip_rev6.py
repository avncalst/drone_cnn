#!/usr/bin/env python3

import threading
import cv2
import depthai as dai
import numpy as np
import time
import sys
import queue
from dataclasses import dataclass
from simple_pid import PID
from pymavlink import mavutil
# Fix for older dronekit versions
import collections
collections.MutableMapping = collections.abc.MutableMapping

from dronekit import connect, VehicleMode

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------- Configuration & State ---------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

@dataclass
class FlightState:
    """Thread-safe container for sharing data between AI and Control threads."""
    # cx: int = 320
    # cy: int = 200
    pix_err: float = 0.0   # Pixel error (difference from target)
    # tyt: float = 200.      # Target Y (the goal height/distance in frame)

    is_tracking: bool = False
    target_id: int = 0
    obstacle_state: str = 'fly'
    per_center_f: float = 0.0

    def __post_init__(self):
        """Initialize mutable objects here so each instance gets its own."""
        self._lock = threading.Lock()
        self.cur_tx: int = 320
        self.cur_ty: float = 200.0

        # PID Calculation for Yaw and Pitch.
        # EK3_RNG_USE_HGT is set to 70% meaning the vertical rangefinder is used
        # for determining the vertical height up to 70% of the max height of the
        # TF02 pro sensor (0.7*13m) provided the vxy is limited to 2 m/sec. If
        # conditions are not met the baro is used for determining the height.

        self.pid_yaw = PID(
            Kp=0.45,          # unchanged — proportional response is well-tuned
            Ki=0.015,         # REV6: reduced from 0.04 → 0.015; log showed integral winding
                              # up during obstacle halts and causing sign-crossing overshoot
                              # on resume.  0.015 still removes steady-state offset in ~8s.
            Kd=0.01,          # unchanged
            setpoint=0.0,
            output_limits=(-0.5, 0.5),
            sample_time=None
        )

        self.pid_pitch = PID(
            Kp=0.020,        # slightly reduced — input is now pre-smoothed so less Kp needed
            Ki=0.0005,       # keep low to avoid slow integral wind-up
            Kd=0.010,        # REDUCED: Kd on a smoothed signal; high Kd on noisy input amplifies noise
            setpoint=200,
            output_limits=(-0.5, 0.5),  # tightened slightly; drone rarely needs full 0.6 m/s
            sample_time=None
        )

        # EMA smoother for cur_ty before it enters the pitch PID.
        # alpha=0.25 → ~4-frame lag at 20 fps, enough to kill single-frame spikes.
        # Increase alpha (→1.0) for faster response; decrease (→0.0) for more smoothing.
        self._EMA_ALPHA = 0.25
        self._ty_ema: float = 200.0   # initialised to setpoint; reset on new tracking target

        # Slew-rate limiter for pitch output: max change per control cycle (0.15 s).
        # 0.18 m/s per step = 1.2 m/s² — smooth enough to avoid the lurch.
        # Raise to 0.30 if responsiveness feels sluggish.
        self._PITCH_SLEW = 0.18
        self._last_pitch: float = 0.0

    def update_tracking(self, cur_tx, cur_ty, pix_err, tracking_status):
        with self._lock:
            self.cur_tx = cur_tx
            # Apply EMA smoothing to cur_ty before storing it.
            # This is what the pitch PID will actually read.
            self._ty_ema = self._EMA_ALPHA * cur_ty + (1.0 - self._EMA_ALPHA) * self._ty_ema
            self.cur_ty = self._ty_ema
            self.pix_err = pix_err
            self.is_tracking = tracking_status

    def reset_ema(self, ty_seed: float):
        """Call when tracking starts fresh so the EMA doesn't ramp up from stale value."""
        with self._lock:
            self._ty_ema = ty_seed
            self._last_pitch = 0.0

    def update_obstacle(self, state, density):
        with self._lock:
            self.obstacle_state = state
            self.per_center_f = density

    def snapshot(self):
        """Atomically read all fields needed by traceDrone."""
        with self._lock:
            return {
                'cur_tx':     self.cur_tx,
                'cur_ty':     self.cur_ty,
                'last_pitch': self._last_pitch,
                'pitch_slew': self._PITCH_SLEW,
            }

    def update_last_pitch(self, value: float):
        """Write back the slew-limited pitch under the lock."""
        with self._lock:
            self._last_pitch = value

# MAVLink Mask constants
X_IGNORE, Y_IGNORE, Z_IGNORE = 1, 2, 4
AX_IGNORE, AY_IGNORE, AZ_IGNORE = 64, 128, 256
YAW_IGNORE = 1024          # ignore yaw angle; use yaw_rate instead
MASK_VELYAWRATE = X_IGNORE | Y_IGNORE | Z_IGNORE | AX_IGNORE | AY_IGNORE | AZ_IGNORE | YAW_IGNORE

IP_SITL = 'udpin:192.168.1.32:15550'
IP_DRONE = 'udpin:127.0.0.1:15550'
FLAG_DRONE = True  # Set to True for physical drone
FRAME_WIDTH = 640
FRAME_HEIGHT = 400
STREAM_FPS = 20

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ------------------------- Drone Utilities -------------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def velyaw(velocity_x, velocity_y, velocity_z, yaw_rate, vehicle):
    """Send a body-frame velocity + yaw-RATE command.

    yaw_rate is in rad/s (positive = nose right / clockwise from above).
    The yaw angle field is set to 0 and ignored via MASK_VELYAWRATE.
    """
    return vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        type_mask=MASK_VELYAWRATE,
        x=0, y=0, z=0,
        vx=velocity_x, vy=velocity_y, vz=velocity_z,  # m/s
        afx=0, afy=0, afz=0,
        yaw=0, yaw_rate=yaw_rate   # rad, rad/s
    )



def traceDrone(shared_state):
    # 1. Atomically grab a consistent snapshot of all shared mutable state.
    snap = shared_state.snapshot()

    # 2. Call PID objects — each call is a single atomic operation on its own
    #    internal state.  GIL protects individual bytecode ops, but a read-
    #    modify-write across two attributes (integral accumulator + last_input)
    #    is NOT atomic.  Serialise access with the lock.

    # In traceDrone(), normalize cur_tx to [-1.0, +1.0] before feeding the PID.
    # Apply a deadband of ±0.05 (~16 pixels) to prevent the yaw PID from
    # chasing sub-pixel noise and winding up the integral over trivial offsets.
    # REV6: widened from ±0.02 to ±0.05 — log showed yaw sign-crossing at step 34
    # when tx_norm was only +0.006, i.e. well inside the old deadband edge where
    # the integral had already driven the output negative.
    DEADBAND = 0.05
    with shared_state._lock:
        tx_normalized = (snap['cur_tx'] - 320) / 320.0
        if abs(tx_normalized) < DEADBAND:
            tx_normalized = 0.0
        yaw_rate  = -shared_state.pid_yaw(tx_normalized)   # rad/s: negative so nose follows target
        raw_pitch =  shared_state.pid_pitch(snap['cur_ty'])

    # 3. Slew-rate limiter — operates entirely on local variables.
    delta         = raw_pitch - snap['last_pitch']
    clamped_delta = max(-snap['pitch_slew'], min(snap['pitch_slew'], delta))
    speed_pitch   = snap['last_pitch'] + clamped_delta

    # 4. Write the new _last_pitch back atomically.
    shared_state.update_last_pitch(speed_pitch)

    return yaw_rate, speed_pitch

def progress(string):
    print(string, file=sys.stdout)
    sys.stdout.flush()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# -------------------------- Pipeline -------------------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def create_oak_pipeline(blob_path):
    pipeline = dai.Pipeline()
    
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
    camRgb.setFps(STREAM_FPS)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    manip.initialConfig.setResize(416, 416) # size needed by mobileNet
    manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p) # The NN model expects BGR input

    # Setting node configs
    stereoDepth.initialConfig.setConfidenceThreshold(255)
    stereoDepth.setSubpixel(True)

    detectionNetwork.setBlobPath(blob_path)
    detectionNetwork.setConfidenceThreshold(0.5)
    detectionNetwork.input.setBlocking(False)
    detectionNetwork.setIouThreshold(0.5)
    detectionNetwork.setNumClasses(20) #voc=20, coco=80
    detectionNetwork.setCoordinateSize(4)
    detectionNetwork.setAnchors([])
    detectionNetwork.setAnchorMasks({})
    detectionNetwork.setNumInferenceThreads(2)
   

    objectTracker.setDetectionLabelsToTrack([14]) # Person
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
    objectTracker.out.link(trackerOut.input)
    camRgb.preview.link(xoutRgb.input)

    return pipeline

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# -------------------------- AI Thread ------------------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def ai_thread_func(frame_queue, shared_state,vehicle,event):
    if FLAG_DRONE:
        blob_path = '/home/pi/drone_exe/drone/models/yolov8n_voc_openvino_2022.1_4shave.blob'
    else:
        blob_path = 'models/yolov8n_voc_openvino_2022.1_4shave.blob'

    labelMap = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                "sofa", "train", "tvmonitor"]
        
    pipeline = create_oak_pipeline(blob_path)
    crit = 10
    start_time = time.monotonic()
    counter = 0
    fps = 0
    tracking_active_prev = False # For capturing target Y once
    #+++++++++++++++++++++++++++
    dist_thres_mm = 2000 # mm
    # dist_thres_mm = 250 # mm # in case of testing
    dist_safe = 200 # mm
    #+++++++++++++++++++++++++++    

    with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:
        q_rgb = device.getOutputQueue("rgb", maxSize=1, blocking=False)
        q_track = device.getOutputQueue("tracklets", maxSize=1, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=1, blocking=False)
        
        while True:
            frame = q_rgb.get().getCvFrame()
            tracklets = q_track.get().tracklets
            depth_frame = q_depth.get().getFrame()

            # Obstacle Logic
            # 2 arrays derived from depthFrame with distances D: 200<D<dist_thres & dist_thres+200 mmm
            dist_zone = cv2.inRange(depth_frame,200,dist_thres_mm) # detect 200<dist(mm)<dist_thres_mm
            dist_fl = cv2.inRange(depth_frame,200,dist_thres_mm+dist_safe) # needed for defining a hysteresis 200mm            
            
            center_win = dist_zone[56:370, 166:469]
            center_win_fl = dist_fl[56:370,166:469]
            density = cv2.countNonZero(center_win) // 100
            density_fl = cv2.countNonZero(center_win_fl) // 100

            # FPS Calculation
            counter += 1
            if (time.monotonic() - start_time) > 1:
                fps = counter / (time.monotonic() - start_time)
                counter = 0
                start_time = time.monotonic()            

            # Tracking Logic
            target_too_close = False
            found_target = False
            current_tx, current_ty = 320, 200
            
            for t in tracklets:
                if t.id == shared_state.target_id:
                    roi = t.roi.denormalize(416, 416)
                    nx = int(roi.topLeft().x * 0.96 + 120)
                    ny = int(roi.topLeft().y * 0.96)
                    mx = int(roi.bottomRight().x * 0.96 + 120)
                    my = int(roi.bottomRight().y * 0.96)
                    if my > 395:
                        target_too_close = True
                    # Draw UI
                    cv2.rectangle(frame, (nx, ny), (mx, my), (0,0,255), 2)
                    cv2.putText(frame, f"{labelMap[t.label]}", (nx+10, ny + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0))
                    cv2.putText(frame, f"ID: {t.id}", (nx+10, ny + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0))

                    # Only update the flight controller if the status is actually stable
                    if t.status.name == "TRACKED":
                        current_tx = (mx-nx)//2 + nx
                        # current_ty = (my-ny)//2 + ny
                        current_ty = (my-ny)
                        found_target = True
                    # break
            # 1. Initialize with the current state to prevent UnboundLocalError
            new_obs_state = shared_state.obstacle_state 

            # 2. Check for state changes
            if density >= crit or target_too_close:
                new_obs_state = 'stop'
            elif density_fl < crit and not target_too_close:
                # Only transition back to 'fly' if the flight path is clear 
                # and the hysteresis threshold (density_fl) is met
                new_obs_state = 'fly'

            # 3. Update the shared state (it will either be the new value or the old one)
            # REV6: if transitioning stop→fly, reset the yaw PID so any integral that
            # wound up during the halt does not fire as a large yaw burst on resume.
            prev_obs_state = shared_state.obstacle_state
            shared_state.update_obstacle(new_obs_state, density)
            if prev_obs_state == 'stop' and new_obs_state == 'fly':
                with shared_state._lock:
                    shared_state.pid_yaw.reset()

            # RC Switch / Event Logic
            ch8 = vehicle.channels.get('8', 0)
            tracking_enabled_by_rc = not FLAG_DRONE or (FLAG_DRONE and ch8 > 1400)

            if tracking_enabled_by_rc and found_target:
                # Capture target Y only on the first frame tracking starts
                if not tracking_active_prev:
                    shared_state.tyt = current_ty
                    shared_state.pid_pitch.setpoint = current_ty
                    # RESET PIDs to clear any built-up integral windup
                    shared_state.pid_yaw.reset()
                    shared_state.pid_pitch.reset()
                    # Seed the EMA at the actual target position so it doesn't
                    # ramp up from a stale value and cause an initial lurch.
                    shared_state.reset_ema(current_ty)
                    tracking_active_prev = True
                
                event.set()
                pixel_error = shared_state.tyt - current_ty
                shared_state.update_tracking(current_tx, current_ty, pixel_error, True)
                col = (0,0,255)
                txt = "ON"                
            else:
                event.clear()
                tracking_active_prev = False
                shared_state.update_tracking(320, 200, 1.0, False)
                col = (0,0,0)
                txt = "OFF"
                if vehicle.mode.name == 'GUIDED':
                    vehicle.mode = VehicleMode("LOITER")

            # HUD
            cv2.rectangle(frame, (120, 0), (520, 400), (0, 255, 0),3)
            cv2.putText(frame, f"Mode: {txt}   Obs: {shared_state.obstacle_state}", (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.6, col)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            frame_queue.put(frame)
            cv2.imshow('Drone View', frame)
            if cv2.waitKey(1) == 27: break

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------- Control Thread ----------------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def slide_thread_func(shared_state, event, vehicle, log_file):
    mes_T = 0.15  # Changed from 1.0 to 0.1 (10Hz) for smoother control
    while True:
        event.wait()
        if vehicle.mode.name != 'GUIDED':
            vehicle.mode = VehicleMode("GUIDED")
            while not vehicle.mode.name == 'GUIDED': time.sleep(0.2)
        
        while event.is_set():
            if shared_state.obstacle_state == "fly":
                if shared_state.is_tracking:

                    sy, sp = traceDrone(shared_state)
                    # Send velocity + yaw-rate command
                    msg = velyaw(sp, 0, 0, sy, vehicle)
                    vehicle.send_mavlink(msg)

                    try:
                        print(f"Tracking: YawRate={sy:.3f} rad/s  Pitch={sp:.2f} m/s", file=log_file)
                        print(f"cur_ty: {shared_state.cur_ty}, pixelerror: {shared_state.pix_err}, tyt: {shared_state.tyt}", file=log_file)
                        print(f"cur_tx: {shared_state.cur_tx}, tx_norm: {(shared_state.cur_tx-320)/320:.3f}", file=log_file)
                        log_file.flush() # Ensure data is written to disk immediately
                    except ValueError:
                        # File was closed by the main thread
                        pass
                else:
                    # Hover if tracking is lost
                    vehicle.send_mavlink(velyaw(0, 0, 0, 0, vehicle))
            else:
                # Stop if obstacle detected
                vehicle.send_mavlink(velyaw(0, 0, 0, 0, vehicle))
                try:
                    print(f"OBSTACLE - HALTING", file=log_file)
                    log_file.flush() # Ensure data is written to disk immediately
                except ValueError:
                    # File was closed by the main thread
                    pass
            
            time.sleep(mes_T)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ------------------------- Target ----- ----------------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def iden_thread_func(shared_state, event):
    while True:
        try:
            val = input('Enter target ID: ')
            with shared_state._lock:
                shared_state.target_id = int(val)
        except EOFError: break
        except: pass

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++ Streaming+++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def sav(frame_queue):
    mediamtx_url = "rtsp://127.0.0.1:8554/stream0"
    
    # HW Accelerated encoder for RPi4
    encoder = (
        f"v4l2h264enc extra-controls=\"controls,video_bitrate=800000\" "
        f"! video/x-h264,profile=baseline,stream-format=byte-stream"
    ) if FLAG_DRONE else "x264enc speed-preset=veryfast tune=zerolatency bitrate=800"

    gst_str = (
        f"appsrc is-live=true block=true "
        f"! video/x-raw,format=BGR,width={FRAME_WIDTH},height={FRAME_HEIGHT},framerate={STREAM_FPS}/1 "
        f"! videoconvert "
        f"! {encoder} "
        f"! rtspclientsink location={mediamtx_url}"
    )

    out = cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 0, STREAM_FPS, (FRAME_WIDTH, FRAME_HEIGHT), True)
    
    if not out.isOpened():
        progress("GStreamer Error: Check MediaMTX or Pipeline string.")
        return

    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        out.write(frame)
    
    out.release()    # RTSP/Saving logic here

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------------- Main --------------------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == "__main__":
    shared_state = FlightState()
    event = threading.Event()
    frame_queue = queue.Queue()

    conn_path = IP_DRONE if FLAG_DRONE else IP_SITL
    progress(f"Connecting to: {conn_path}")
    
    try:
        # source_system=1, source_component=10 often helps with RPi communication
        vehicle = connect(conn_path, wait_ready=False, source_system=1, source_component=10)
        progress("Vehicle Connected!")
    except Exception as e:
        progress(f"Connection Failed: {e}")
        sys.exit()    

    vehicle.wait_ready('autopilot_version')

    with open('oak_log.txt', 'w') as f:
        t_ai = threading.Thread(target=ai_thread_func, args=(frame_queue, shared_state, vehicle, event), daemon=True)
        t_slide = threading.Thread(target=slide_thread_func, args=(shared_state, event, vehicle, f), daemon=True)
        t_iden = threading.Thread(target=iden_thread_func, args=(shared_state, event), daemon=True)
        t_sav = threading.Thread(target=sav, args=(frame_queue,), daemon=True)

        t_ai.start()
        t_slide.start()
        t_iden.start()
        t_sav.start()

        t_ai.join()
