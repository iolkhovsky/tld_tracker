## TLD image tracker

This project implements object tracking algorithm described in http://vision.stanford.edu/teaching/cs231b_spring1415/papers/Kalal-PAMI.pdf. Source paper added to docs folder.
C++ implementation uses OpenCV library for basic image processing operations and I/O.

Library and demo application have been tested under Ubuntu 18.04 and OpenCV 4.0.0

### Building
Clone repo and run building script in the repo root - build_tld. After that built shared library with TLD sources will appear in build-tldlibrary directory and build demo application will appear in build-tlddemoapp directory.

### How to run
You will see help information if run demo application with option --help:
	./tld_demo_app --help

Example of running with web-camera video stream:
	./tld_demo_app --webcam --camid=DEVICE_ID
Example of running with video stream read from videofile:
	./tld_demo_app --video --videopath=ABS_PATH_TO_VIDEO

After application will start press 't' key to designate target. Select bounding box by mouse and press space key to start tracking.
