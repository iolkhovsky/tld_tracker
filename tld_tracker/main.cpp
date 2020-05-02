/*#include "mainwindow.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}*/

#include <iostream>
#include <tracker/tld_tracker.h>
#include <tracker/utils.h>
#include <profile.h>

using namespace std;
using namespace TLD;
using namespace cv;

int main(int argc, char** argv) {

    Settings s;
    TldTracker tracker(s);

    cv::VideoCapture cap;
    cap = VideoCapture(0);
    if(!cap.isOpened()) {
      cout << "Error opening web-camera " << endl;
      return 0;
    }

    cv::Mat src_frame, frame, gray;
    Rect target, current;
    Candidate state;
    while(cap.isOpened()) {
        LOG_DURATION("Processing");

        cap >> src_frame;
        resize(src_frame, frame, Size(640, 480));
        if (frame.empty())
            break;
        cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        state = tracker.SetFrame(gray);

        imshow("Stream", frame);

        // quit on x button
        char in_symbol = waitKey(1);
        if  (in_symbol == 'q')
            break;
        else if (in_symbol == 't') {
            target = selectROI("Stream", frame);
            tracker.StartTracking(target);
            cout << "Target: " << target;
        }

   }

    return 0;
}
