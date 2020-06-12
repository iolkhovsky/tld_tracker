## TLD image tracker

This project implements object tracking algorithm described in http://vision.stanford.edu/teaching/cs231b_spring1415/papers/Kalal-PAMI.pdf. Source paper added to docs folder.
C++ implementation uses OpenCV library for basic image processing operations and I/O.

Library and demo application have been tested under Ubuntu 18.04 and OpenCV 4.0.0

![Alt text](attachments/tld_demo.gif?raw=true "Demo application")


### Building
Clone repo and run building script (it runs cmake + make) in the repo root - build_tld. 
After process will complete built shared library with TLD sources will appear in the build-tldlibrary directory and build demo application will appear in the build-tlddemoapp directory.
![Alt text](attachments/tld_build_1.png?raw=true "Building example 1")
![Alt text](attachments/tld_build_2.png?raw=true "Building example 2")


### How to run
You will see help information if run demo application with option --help:
	./tld_demo_app --help

![Alt text](attachments/tld_help_and_test.png?raw=true "Help & test")

Example of application launch with web-camera video stream:
	./tld_demo_app --webcam --camid=DEVICE_ID
Example of application launch with video stream read from videofile:
	./tld_demo_app --video --videopath=ABS_PATH_TO_VIDEO

After application will start press 't' key to designate target. Set designation bounding box by mouse and press space key to start tracking.
