# drone_cnn
* python cnn files for drone obstacles avoidance.
* files include training cnn (avcNet folder) and files for running cnn on rpi3 (rpi3 folder)
* example cnn trained binary files (hdf5 - pb) included in rpi3 folder
* different interference methods added: OpenVINO (Intel), DNN-OpenCV, tensorflow, keras.
* on rpi3 DNN-OpenCV: 6 fps, DNN-MYRIAD (NCS1): 20 fps using OpenCV version 4.1.0
* a simple python obstacle avoidance script is included
* Wiki: [brief project description](https://github.com/avncalst/drone_cnn/wiki) describing 2 approaches for cnn development
* donkeycar folder added containing the modified/added files making ArduPilot Rover-copter compatible with donkeycar
* Wiki ros paragraph added for ArduCopter sitl simulations
* autoencoder folder added
* folder transfer learning jupyter notebooks added: custom object detection and custom classification; this folder also includes an application using mobilenet ssd face tracking to control a DJI TELLO drone. 
* revisions: an ai_avoid_rev.py is added in rpi3-rpi4 folder using tflite inference with coral usb accelerator; the training is done by transfer learning using a jupyter notebook on Google's Colab (see folder transfer_learning).
* a real_avoid.py file is added in rpi3-rpi4 using the Intel RealSense camera for obstacle avoidance
* rosetta folder added containing python scripts using the rosetta app to control DJI drones
* oak_avoid.py file added in rpi3-rpi4 using OAK-D depthAI
* oak_avoid_rev3.py file added in rpi3-rpi4 containing a new avoidance algorithme and person tracking

![variational autoencoder](https://github.com/avncalst/drone_cnn/blob/master/images/test29.png)

