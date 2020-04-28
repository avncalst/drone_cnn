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

![variational autoencoder](https://github.com/avncalst/drone_cnn/blob/master/images/test29.png)

