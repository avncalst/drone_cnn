#!/usr/bin/env python

import rospy
import cv2
import imutils

from sensor_msgs.msg        import Image
from cv_bridge              import CvBridge, CvBridgeError
from operator               import itemgetter


"""
ON PC: roslaunch roslaunch gazebo_ros empty_world.launch
   0------------------> x (cols) Image Frame
   |
   |        c    Camera frame
   |         o---> x
   |         |
   |         V y
   |
   V y (rows)
SUBSCRIBES TO:
    /iris/camera/image_raw: Source image topic
    
PUBLISHES TO:

"""
k = 0.66
fly_1=0.9
state='fly'
label='forward'
proba=1

def callback(data):
    #--- Assuming image is 208x208
    bridge = CvBridge()
    global k
    global fly_1
    global state
    global label
    global proba

    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
    # use cv2.dnn module for inference
    net=cv2.dnn.readNetFromTensorflow('/home/avncalst/Dropbox/donkeycar/mycar/models/fly1.pb')
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) # if no NCS stick
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) # if no NCS stick
    
    resized=cv2.resize(cv_image, (64, 64))
    scale=1/255.0 # AI trained with scaled images image/255
    blob = cv2.dnn.blobFromImage(resized,scale,
                (64, 64), (0,0,0),swapRB=False,ddepth = 5)
    net.setInput(blob)
    predictions = net.forward()
    # print(predictions)
    
    #===========================================================================

    # classify the input image
    fly = predictions[0][0]
    # build the label

    my_dict = {'stop':predictions[0][3], 'left':predictions[0][1],
        'right':predictions[0][2]}        
    maxPair = max(my_dict.items(), key=itemgetter(1))
    fly_f = k*fly_1 + (1-k)*fly
    fly_1 = fly_f


    if state == 'avoid':
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
    output = imutils.resize(cv_image, width=208)
    cv2.putText(output, label_1, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)



    # show the output frame
    cv2.imshow("Frame", output)
    key = cv2.waitKey(10) & 0xFF
 
    # # if the `Esc` key was pressed, destroy windows
    # if key == 27:
    #     cv2.destroyAllWindows()

def subscriber():
    rospy.Subscriber("/iris/camera/image_raw", Image, callback)
    print ("<< Subscribed to topic /iris/camera/image_raw")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("Shutting down")
        

if __name__ == "__main__":
    rospy.init_node("avctech")
    subscriber()