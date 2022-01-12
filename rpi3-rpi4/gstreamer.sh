#!/bin/bash
gst-launch-1.0 ximagesrc xname=Frame ! video/x-raw,framerate=20/1 ! videoscale method=0 ! video/x-raw,width=640,height=400 ! videoconvert ! omxh264enc control-rate=3 target-bitrate=500000 ! filesink location=/dev/ttyAMA1 
