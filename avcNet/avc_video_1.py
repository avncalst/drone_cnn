from keras.preprocessing.image import img_to_array
from keras.models import load_model
##from dronekit import connect, VehicleMode
import imutils
import argparse
import numpy as np
import time
import cv2
import tensorflow as tf
from operator import itemgetter


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to video file")
args = vars(ap.parse_args())

print("[INFO] start video...")
cap = cv2.VideoCapture(args["video"])
##cap = cv2.VideoCapture("IA_video.avi")
##out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (400,300))
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc(*'XVID'), 25, (400,300))


# load the trained convolutional neural network
print("[INFO] loading network...")
##model = load_model("./avcnet_ref.model")
model = load_model("./avcnet_best_1.hdf6",custom_objects={"tf": tf})
# model = load_model("./avcnet_resnet.model")

# default parameters
alfa=.1
dist=300
dist_old=300

while True:
    _,frame = cap.read()
    frame = imutils.resize(frame, width=400)
##    H, S, V = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
##    eq_V = cv2.equalizeHist(V)
##    frame = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR)    

##    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   
    orig  = frame.copy()

    
    
##    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
##    frame = cv2.equalizeHist(frame)    
    frame = cv2.resize(frame, (64,64))
    frame = frame.astype("float")/255.0
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)


    # classify the input image
    (stop, left,right,fly) = model.predict(frame)[0]
                  
    # build the label
##    lijst=[stop,left,right,fly]
##    print lijst
    my_dict = {'stop':stop, 'left':left, 'right':right,'fly':fly}
    print my_dict
    maxPair = max(my_dict.iteritems(), key=itemgetter(1))
    label=maxPair[0]
    proba=maxPair[1]
##    idx=lijst.index(max(lijst))
##    print idx, lijst[idx]
##    if idx==0:
##        label="stop"
##    if idx==1:
##        label="left"
##    if idx==2:
##        label="right"
##    if idx==3:
##        label="fly"
##
##    proba = lijst[idx]
    label = "{} {:.1f}%".format(label, proba * 100)

    # draw the label on the image
    output = imutils.resize(orig, width=400)
    cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)
    
    # show the output frame
    cv2.imshow("Frame", output)
    key = cv2.waitKey(10) & 0xFF

    # Write the frame into the file 'output.avi'
    out.write(output)
    

    # if the `Esc` key was pressed, break from the loop
    if key == 27:
        break



# do a bit of cleanup
cv2.destroyAllWindows()
##cap.stop()
