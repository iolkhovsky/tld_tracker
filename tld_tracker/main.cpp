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
#include <profile.h>

using namespace std;
using namespace TLD;
using namespace cv;

int main(int argc, char** argv) {

    cout << "System arguments have been received: " << endl;
    for (int i = 0; i < argc; i++) {
        cout << "#" << i << ": " << argv[i] << endl;
    }

    TldTracker tracker = make_tld_tracker();

    cv::VideoCapture cap;
    cap = VideoCapture(0);
    if(!cap.isOpened()) {
      cout << "Error opening web-camera " << endl;
      return 0;
    }

    cv::Mat src_frame, frame, gray;
    Rect target, current;
    Candidate result;
    while(cap.isOpened()) {
        LOG_DURATION("Processing")

        cap >> src_frame;
        resize(src_frame, frame, Size(640, 480));
        if (frame.empty())
            break;
        cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        result = tracker.SetFrame(gray);
        std::vector<TLD::Candidate> _det_proposals;
        std::vector<TLD::Candidate> _det_clusters;
        TLD::Candidate _track_proposal;
        std::tie(_det_proposals, _det_clusters, _track_proposal) = tracker.GetProposals();

        auto deb_frame = frame.clone();

        TLD::drawCandidate(frame, result);
        TLD::drawCandidates(deb_frame, _det_clusters);
        TLD::drawCandidate(deb_frame, _track_proposal);

        imshow("Stream", frame);
        imshow("Debug", deb_frame);

        // quit on x button
        auto in_symbol = waitKey(1);
        if  (in_symbol == 'q')
            break;
        else if (in_symbol == 't') {
            target = selectROI("Stream", frame);
            tracker.StartTracking(target);
        }

        std::cout << tracker;

   }

    return 0;
}


//int main(int argc, char *argv[])
//{
//    vector<FernFeatureExtractor> _feat_extractors;
//    vector<ScanningGrid> _scanning_grids;
//    vector<ObjectClassifier<uint16_t, 1024>> _classifiers;

//    cv::Rect _designation(50, 60, 100, 100);

//    _feat_extractors.clear();
//    _scanning_grids.clear();
//    _classifiers.clear();
//    for (auto i = 0; i < 1; i++) {
//        _scanning_grids.emplace_back(ScanningGrid({640, 480}));
//        _scanning_grids.back().SetBase({_designation.width, _designation.height}, 0.1, {0.9,1.0,1.1});
//        _feat_extractors.push_back(_scanning_grids.back());
//        _classifiers.emplace_back(ObjectClassifier<BinaryDescriptor, BINARY_DESCRIPTOR_CNT>());
//    }

//    auto i = _classifiers.front().Predict(100);
//    cout << i << endl;
//    return 0;
//}


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
