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
        self.vehicle = None
        self.connect()
        self.arducopter = True # ardurover = not arducopter


    def msg_sensor(self,dist,orient,distmax):
    
        msg = self.vehicle.message_factory.distance_sensor_encode(
            0,      # time system boot, not used
            10,     # min disance cm
            distmax,# max dist cm
            dist,   # current dist, int
            0,      # type sensor, laser
            1,      # on board id, not used
            orient, # orientation: 0...7, 25
            0,      # covariance, not used
            )
    
        self.vehicle.send_mavlink(msg)

        
    def connect(self,flag=[]):
        # connect vehicle to access point
        if flag==[]:
##            self.vehicle = connect('udpout:192.168.42.1:14550', wait_ready=False) # mavproxy listens udpin 0.0.0.0:14550
            self.vehicle = connect('udpin:127.0.0.1:15550', wait_ready=False)
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

            flag.append('not empty')

        else:
            print('already connected')
            pass

        
    
    

class MonitorArdupilot(car):
    def __init__(self, class_car): 
        self.vehicle = class_car.vehicle
        self.arducopter = class_car.arducopter
        self.running = True
        self.recording = False
        self.mode='user'
        self.angle=0
        self.throttle=0
        

    def run_threaded(self, img_arr=None):
        self.img_arr = img_arr
        return self.angle, self.throttle, self.mode, self.recording

    def update(self):
        if self.arducopter:

            while self.running:
                pitch = self.vehicle.channels['2']      # copter (throttle)
                roll = self.vehicle.channels['1']       # copter (angle)
                mod = self.vehicle.channels['6']        # fly, stop, shift left, right
                flag_rec = self.vehicle.channels['7']   # drive - record mode
                roll= (roll-1500)/500.0                 # scaling inputs (-1,1)
                pitch = -(pitch-1500)/500.0             # scaling inputs (-1,1)
                self.throttle = pitch
                if flag_rec > 1400 and flag_rec < 1700:
                    flag_record = True
                else:
                    flag_record = False


                # angle values will result in np.array with dim 4 @ training
                if mod < 1400 and flag_record:
                    self.recording = True
                    self.angle = -0.9 # fly
                    # print('fly')
                elif (mod > 1400 and mod < 1700) and flag_record:
                    rl = roll<-0.03
                    rr = roll>0.03
                    po = pitch>-0.03 and pitch<0.03
                    self.recording = (po and rl) or (po and rr)
                    # print('roll,pitch:',rl,rr,po)
                    # print('recording:',self.recording)
                    if po and rl:
                        self.angle = -0.3
                        # print('left')
                    elif po and rr:
                        self.angle = 0.3
                        # print('right')
                elif mod > 1700 and flag_record:
                    self.recording = True
                    self.angle = 0.9
                    # print('stop')

                time.sleep(0.05) # needed to avoid false records (angle=throttle=0)                
            
                self.recording = False
                self.angle = 0
                self.throttle = 0

        else:

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


    def shutdown(self):
        self.running = False
        time.sleep(0.5)


class DriveArdupilot(car):
    def __init__(self, class_car):
        self.vehicle = class_car.vehicle
        self.arducopter = class_car.arducopter
        

    def run(self, angle, throttle):
        if self.arducopter:
            pass
        else:

            # if self.vehicle == None:
            #     print ('no connection, connecting ...')
            #     self.connect()
            # else:
            #     pass
                    
            angle = int(angle*500.0 + 1500)         #scaling (-1,1) -> (1500,2000)
            throttle = int(throttle*500.0 +1500)     #scaling (-1,1) -> (1500,2000)
            self.vehicle.channels.overrides['1']=angle
            self.vehicle.channels.overrides['3']=throttle

    def shutdown(self):
        # self.vehicle.channels.overrides['2']=1500  # stop driving
        self.vehicle.channels.overrides['3']=1500  # stop driving
        time.sleep(0.5)
