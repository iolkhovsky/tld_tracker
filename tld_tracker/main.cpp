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
#include <tracker/tld_utils.h>
#include <profile.h>

using namespace std;
using namespace TLD;
using namespace cv;

int main(int argc, char** argv) {

    cout << "System arguments have been received: " << endl;
    for (int i = 0; i < argc; i++) {
        cout << "#" << i << ": " << argv[i] << endl;
    }

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
        LOG_DURATION("Processing")

        cap >> src_frame;
        resize(src_frame, frame, Size(640, 480));
        if (frame.empty())
            break;
        cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        state = tracker.SetFrame(gray);
        std::vector<TLD::Candidate> _det_proposals;
        TLD::Candidate _track_proposals;
        std::tie(_det_proposals, _track_proposals) = tracker.GetProposals();
        TLD::drawCandidate(frame, _track_proposals);
        TLD::drawCandidates(frame, _det_proposals);

        imshow("Stream", frame);

        // quit on x button
        auto in_symbol = waitKey(1);
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

//int main() {

//    TranformPars pars;
//    pars.angles = {-15, 0, 15};
//    pars.scales = {0.5, 1.0, 1.5};
//    pars.translation_x = {0};
//    pars.translation_y = {0};
//    pars.overlap = 0.1;

//    cv::Rect roi(200, 200, 150, 150);

//    cv::Mat src = imread("Lenna.jpg");
//    cvtColor(src, src, cv::COLOR_BGR2GRAY);

//    imshow("src", src);

//    {
//        LOG_DURATION("Augmentation")
//        Augmentator augm(src, roi, pars);
//        int i = 0;
//        for (auto subframe: augm.SetClass(ObjectClass::Positive)) {
//            cout << i++ << endl;
//            stringstream ss("p");
//            ss << i;
//            imshow(ss.str().c_str(), subframe);
//        }
//        waitKey(0);
//        i = 0;
//        for (auto subframe: augm.SetClass(ObjectClass::Negative)) {
//            cout << i++ << endl;
//            stringstream ss("n");
//            ss << i;
//            imshow(ss.str().c_str(), subframe);
//            if (i > 100)
//                break;
//        }
//        waitKey(0);
//    }

//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}
