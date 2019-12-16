from dronekit import connect, VehicleMode
from pymavlink import mavutil 

import time

class car(object):

    '''
    uses Ardupilot to obtain the steering values of the vehicle and steering of the vehicle
    '''    

    def __init__(self):

        print('start Ardupilot ...')
        self.angle = 0.0
        self.throttle = 0.0
        self.mode = 'user'
        self.recording = False
        self.running = True
        self.vehicle = None
        self.connect()


        
    def connect(self,flag=[]):
        # connect vehicle to access point
        if flag==[]:
            self.vehicle = connect('udpout:192.168.42.1:14550', wait_ready=False) # mavproxy listens udpin 0.0.0.0:14550
            # self.vehicle = connect('udpin:127.0.0.1:15550', wait_ready=False)
            self.vehicle.initialize(8,30)
            self.vehicle.wait_ready('autopilot_version')
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

            flag.append('not empty')

        else:
            print('already connected')
            pass

        
    
    

class MonitorArdupilot(car):
    def __init__(self, class_car): 
        self.vehicle = class_car.vehicle
        self.running = True 

    def run_threaded(self, img_arr=None):
        self.img_arr = img_arr
        return self.angle, self.throttle, self.mode, self.recording

    def update(self):
        while self.running:
            angle = self.vehicle.channels['1']
            throttle = self.vehicle.channels['3']
            self.angle= (angle-1500)/500.0           # scaling inputs (-1,1)
            self.throttle = (throttle-1500)/500.0    # scaling inputs (-1,1)
            if self.vehicle.channels['7'] > 1400 and self.vehicle.channels['7'] < 1700:
                self.recording = True
            else:
                self.recording = False
            if self.vehicle.channels['7'] > 1700:
                self.mode = 'local' # mode: 'user', 'local_angle' (only angle), 'local' (angle + throttle)
                # print(self.mode)
            else:
                self.mode='user'
                
                
            # print('\nangle,throttle:',self.angle,self.throttle)
            # print('record:',self.recording) 

    def shutdown(self):
        self.running = False
        time.sleep(0.5)


class DriveArdupilot(car):
    def __init__(self, class_car): 
        self.vehicle = class_car.vehicle     

    def run(self, angle, throttle):
        # if self.vehicle == None:
        #     print ('no connection, connecting ...')
        #     self.connect()
        # else:
        #     pass
                    
        angle = int(angle*500.0 + 1500)         #scaling (-1,1) -> (1500,2000)
        throttle = int(throttle*500.0 +1500)     #scaling (-1,1) -> (1500,2000)
        self.vehicle.channels.overrides['1']=angle
##        self.vehicle.channels.overrides['3']=throttle
        # print(angle)

    def shutdown(self):
        self.vehicle.channels.overrides['3']=1500  # stop driving
        time.sleep(0.5)
