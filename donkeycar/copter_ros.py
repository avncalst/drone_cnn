#!/usr/bin/env python

import roslaunch
import cv2
import imutils
import threading
import time

from operator               import itemgetter
from dronekit               import connect, VehicleMode
from pymavlink              import mavutil 


class copter(object):

    '''
    uses Ardupilot-ROS-Gazebo to obtain the steering values of the vehicle and steering of the vehicle
    '''        
    def __init__(self):
        print ('start Ardupilot-ROS-Gazebo')
        self.angle = 0.0
        self.throttle = 0.0
        self.mode = 'user'
        self.recording = False
        self.vehicle = None
        
        self.arducopter = True # ardurover = not arducopter

        # print('roslaunch')
        # uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        # roslaunch.configure_logging(uuid)
        # launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/avncalst/catkin_ws/src/pilot/launch/iris_world.launch"])
        # launch.start()
          
        self.connect()

        


    def connect(self):
        # connect vehicle to access point
    
        ##vehicle = connect('udpin:192.168.1.61:14550', wait_ready=False)
        self.vehicle = connect('udpin:127.0.0.1:14550', wait_ready=False)
        self.vehicle.initialize(8,30)
        self.vehicle.wait_ready('autopilot_version')
        # Set avoidance enable, proximity type mavlink
        self.vehicle.parameters['avoid_enable']=0
        self.vehicle.parameters['PRX_TYPE']=0
        # Get all vehicle attributes (state)
        print("\nGet all vehicle attribute values:")
        print(" Autopilot Firmware version: %s" % self.vehicle.version)
        print("   Major version number: %s" % self.vehicle.version.major)
        print("   Minor version number: %s" % self.vehicle.version.minor)
        print("   Patch version number: %s" % self.vehicle.version.patch)
        print("   Release type: %s" % self.vehicle.version.release_type())
        print("   Release version: %s" % self.vehicle.version.release_version())
        print("   Stable release?: %s" % self.vehicle.version.is_stable())
        print(" Attitude: %s" % self.vehicle.attitude)
        print(" Velocity: %s" % self.vehicle.velocity)
        print(" GPS: %s" % self.vehicle.gps_0)
        print(" Flight mode currently: %s" % self.vehicle.mode.name)
        # parameter 0: not used
        print('avoid_enable: %s' % self.vehicle.parameters['avoid_enable'])
        print('proximity_type: %s' % self.vehicle.parameters['PRX_TYPE'])
        



class MonitorArdupilot(copter):

    def __init__(self, class_car):
        self.vehicle = class_car.vehicle
        self.arducopter = class_car.arducopter
        self.running = True
        self.recording = False
        self.mode = 'user'
        self.angle = 0
        self.throttle = 0


    def run_threaded(self, img_arr=None):
        self.img_arr = img_arr
        return self.angle, self.throttle, self.mode, self.recording

    def update(self):
        pitch = 1500
        roll = 1500
        throttle = 1000
        mod = 1000
        flag_rec = 1000

        while self.running:
            # in case of Sitl only decorator on_message works to access channels pitch, roll, ...
            @self.vehicle.on_message('RC_CHANNELS')
            def chin_listener(self, name, message):
                # print '%s attribute is: %s' % (name, message)
                nonlocal pitch
                nonlocal roll
                nonlocal throttle
                nonlocal mod
                nonlocal flag_rec
    
                pitch=message.chan2_raw    # copter (throttle)
                roll=message.chan1_raw     # copter (angle)
                throttle=message.chan3_raw # copter (throttle) 
                mod=message.chan6_raw      # fly, stop, shift left, right, switch SD
                flag_rec=message.chan7_raw # drive - record mode, switch SG
                # print ('channels:',pitch,roll,mod,flag_rec)


            if flag_rec > 1400 and flag_rec < 1700:
                flag_record = True
            else:
                flag_record = False

            # angle values will result in np.array with dim 4 @ training
            if mod < 1400 and flag_record:
                self.recording = True
                self.angle = -0.9 #fly
                print('fly')
            elif (mod > 1400 and mod < 1700) and flag_record:
                rl = roll<1460
                rr = roll>1540
                po = pitch>1460 and pitch<1540
                self.recording = (po and rl) or (po and rr)
                # print'roll,pitch:',rl,rr,po
                # print'recording:',recording
                if po and rl:
                    self.angle = -0.3
                    print('left')
                elif po and rr:
                    self.angle = 0.3
                    print('right')
            elif mod > 1700 and flag_record:
                self.recording = True
                self.angle = 0.9
                print('stop')

            time.sleep(0.05) # needed to avoid false records (angle=throttle=0)

            self.recording = False
            self.angle = 0
            self.throttle = 0

    def shutdown(self):
        self.running = False
        time.sleep(0.5)

     



