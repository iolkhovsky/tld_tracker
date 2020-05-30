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
#include <test_runner.h>

using namespace std;
using namespace TLD;
using namespace cv;

void TestFeatureExtractor() {
    cv::Mat color_img = cv::imread("../Lenna.jpg");
    cv::Mat gray, filtered_frame;
    cv::cvtColor(color_img, gray, cv::COLOR_BGR2GRAY);
    cv::blur(gray, filtered_frame, cv::Size(7,7));

    cv::Rect designation;
    designation.x = 100;
    designation.y = 150;
    designation.width = 233;
    designation.height = 172;

    cv::Size imsz(gray.cols, gray.rows);
    auto grid = std::make_shared<ScanningGrid>(imsz);
    TLD::FernFeatureExtractor fext(grid);
    auto scales = grid->GetScales();

    std::vector<cv::Size> positions_per_scale = grid->GetPositionsCnt();
    size_t scale_id = 0;
    for (auto positions: positions_per_scale) {
        double abs_scale = scales.at(scale_id);
        for (auto y_i = 0; y_i < positions.height; y_i++) {
            for (auto x_i = 0; x_i < positions.width; x_i++) {

                cv::Rect strobe;
                strobe.x = x_i * static_cast<int>(abs_scale * grid->GetOverlap().width);
                strobe.y = y_i * static_cast<int>(abs_scale * grid->GetOverlap().height);
                strobe.width = static_cast<int>(abs_scale * designation.width);
                strobe.height = static_cast<int>(abs_scale * designation.height);

                auto subframe = filtered_frame(strobe);

                auto desc_0 = fext(filtered_frame, {x_i, y_i}, scale_id);
                auto desc_1 = fext.GetDescriptor(subframe);
                auto desc_2 =fext.GetDescriptor(gray, strobe);

                ASSERT_EQUAL(desc_0, desc_1)
                ASSERT_EQUAL(desc_1, desc_2)
                ASSERT_EQUAL(desc_0, desc_2)
            }
        }
    }
}

int main(int argc, char** argv) {

    cout << "System arguments have been received: " << endl;
    for (int i = 0; i < argc; i++) {
        cout << "#" << i << ": " << argv[i] << endl;
    }

    TestRunner tr;
    RUN_TEST(tr, TestFeatureExtractor);

    cout << "Enter 'y' to run application" << endl;
    string key;
    cin >> key;
    if (key!="y")
        return 0;

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
