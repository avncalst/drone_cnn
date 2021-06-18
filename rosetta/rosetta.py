import numpy as np
import cv2
import time
# print(cv2.getBuildInformation())

from mss import mss
from PIL import Image
from pynput.mouse import Listener

#===============================def functions================================================
def goto_position_target_local_ned(north, east, down):
    """
    Send SET_POSITION_TARGET_LOCAL_NED command to request the vehicle fly to a specified
    location in the North, East, Down frame.
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, # frame
        0b0000111111111000, # type_mask (only positions enabled)
        north, east, down,
        0, 0, 0, # x, y, z velocity in m/s  (not used)
        0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)
    # send command to vehicle
    vehicle.send_mavlink(msg)  


#===============================def test=====================================================
def mov_mini():
    txt = input("if take-off, hit key to continue")
    print('text:',txt)
    if vehicle.armed==True:
        #-------------------------------------------------
        if txt == 't':
            # vehicle.mode = VehicleMode("GUIDED")
            # while vehicle.mode!='GUIDED':
            #     print("Waiting for drone to enter GUIDED flight mode")
            #     time.sleep(1)

            counter=0
            while counter<5:
                goto_position_target_local_ned(0,0.8,0)
                time.sleep(2)
                print("Moving right")
                counter=counter+1

            time.sleep(2)

            counter=0
            while counter<5:
                goto_position_target_local_ned(0,-0.8,0)
                time.sleep(2)
                print("Moving left")
                counter=counter+1

            time.sleep(2)

            print('landing')
            vehicle.mode=VehicleMode("LAND")
            while not vehicle.mode.name=='LAND':
                print('Waiting for mode LAND')
                time.sleep(0.5)

        #-------------------------------------------------
        else:
            x = 0 # m/s
            z = 0
            while True:
                event.wait()
 
                flight_mode = vehicle.mode.name
                print('flight_mode @ object avoidance:',flight_mode)
                count = 0
                while event.is_set():
                    print('y:',y)
                    goto_position_target_local_ned(x, y, z)
                    time.sleep(2)
                    count +=1

                    if count >= 10:
                        print('landing')
                        vehicle.mode=VehicleMode("LAND")
                        while not vehicle.mode.name=='LAND':
                            print('Waiting for mode LAND')
                            time.sleep(0.5)
                        break

                print('move stopped')


#===============================def capture rec==============================================
print('INFO: draw rectangle')
rec = [] # xy left upper corner,xy right lower corner
def on_click(x, y, button, pressed):
    if pressed:
        print ('Mouse clicked at ({0}, {1}) with {2}'.format(x, y, button))
        rec.append(x)
        rec.append(y)
        print('rec:',rec)
        if len(rec)==4:
            listener.stop()

listener = Listener(on_click=on_click)
listener.start()
listener.join()

#===============================grab image===================================================
def ai():
    import tflite_runtime.interpreter as tflite
    from imutils.video import FPS
    from pycoral.adapters import common
    from operator import itemgetter

    interpreter = tflite.Interpreter('tflite/w_tflite_model_edgetpu.tflite',
        experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("INFO: video recording")
    # out = cv2.VideoWriter('avcrosetta.avi',cv2.VideoWriter_fourcc(*'XVID'), 25, (240,136))
    out = cv2.VideoWriter('avcrosetta.avi',cv2.VideoWriter_fourcc(*'XVID'), 25, (320,240))
    fly_1=0.9
    state='fly'
    global y
    y=0
    k=0.66 # k=(tau/T)/(1+tau/T) tau time constant LPF, T period
    fps = FPS().start()
    sct = mss() # grab video GCS
    w = rec[2]-rec[0]
    h = rec[3]-rec[1]
    # print('w,h:',w,h)
    monitor = {'top': rec[1], 'left': rec[0], 'width': w, 'height': h}
    while True:
        img = Image.frombytes('RGB', (w,h), sct.grab(monitor).rgb)
        frame = np.array(img) # convert PIL image to opencv
        orig  = frame.copy()
        frame = cv2.resize(frame, (224,224))
        frame = frame.astype("float")/255.0 # image is normalized
        frame = np.expand_dims(frame, axis=0)
        frame = frame.astype('float32')  # float32
        interpreter.set_tensor(input_details[0]['index'], frame)
        common.set_input(interpreter, frame)
        interpreter.invoke()
        output_details = interpreter.get_output_details()[0]
        fly = interpreter.tensor(output_details['index'])().flatten()[0]
        left = interpreter.tensor(output_details['index'])().flatten()[1]
        right = interpreter.tensor(output_details['index'])().flatten()[2]
        stop = interpreter.tensor(output_details['index'])().flatten()[3]
        fly_f = k*fly_1 + (1-k)*fly
        fly_1 = fly_f
        my_dict = {'stop':stop, 'left':left, 'right':right}        
        maxPair = max(my_dict.items(), key=itemgetter(1))
        if state == 'avoid':
            # label=maxPair[0] #change label only in fly mode
            # proba=maxPair[1]
            if fly_f*100 >= 60:
                state='fly'
         
        else:
            label='forward'
            proba=fly
            if fly_f*100 <= 50:
                label=maxPair[0]
                proba=maxPair[1]
                state='avoid'

        label_1 = "{} {:.1f}% {}".format(label, proba * 100, state)
        # draw the label on the image
        cv2.putText(orig, label_1, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
        
        # show the output frame
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        out.write(cv2.resize(orig, (320,240)))
        cv2.imshow('Frame', orig)
        
        if state == 'fly':
            event.clear()

        if state == 'avoid':
            event.set()

            if label == 'left':
                y = -0.5
            if label == 'right':
                y = 0.5
            if label == 'stop':
                y = 0


        fps.update()
        if cv2.waitKey(25) & 0xFF == 27:
            cv2.destroyAllWindows()
            out.release() 
            break
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        
#===============================dronekit test================================================
print('INFO: starting dronekit')
from dronekit import connect, VehicleMode
from pymavlink import mavutil 
import threading

# vehicle = connect('udpin:192.168.1.30:14550', wait_ready=False)
vehicle = connect('udpin:192.168.42.14:14550', wait_ready=False)
vehicle.initialize(8,30)
vehicle.wait_ready('autopilot_version')

print("Attitude: %s" % vehicle.attitude)
print('pitch: %s' %vehicle.attitude.pitch)
print("Velocity: %s" % vehicle.velocity)
print("GPS: %s" % vehicle.gps_0)
print("Flight mode currently: %s" % vehicle.mode.name)
print('battery level: %s' % vehicle.battery.level)
print('status: %s'% vehicle.system_status.state)
print('armed: %s'% vehicle.armed)
print('heading: %s' %vehicle.heading)
print('speed: %s'% vehicle.groundspeed)
print('gps fix: %s'% vehicle.gps_0.fix_type)
print('sattelites: %s' %vehicle.gps_0.satellites_visible)
print('location global lat: %s'% vehicle.location.global_frame.lat)
print('location global lon: %s'% vehicle.location.global_frame.lon)
print ('location global alt: %s' %vehicle.location.global_frame.alt)
print('heading: %s'% vehicle.heading)
print('groundspeed: %s' % vehicle.groundspeed)
print('velocity: %s'% vehicle.velocity[2])


event = threading.Event()
t1 = threading.Thread(target=ai)
t2 = threading.Thread(target=mov_mini)
t2.daemon=True # has to run in console
t1.start()
t2.start()
t1.join()
