#include <iostream>
#include <algorithm>

#include "cmdline_parser.h"
#include "tld/tld_tracker.h"
#include "unit_tests.h"
#include "profile.h"

using namespace std;
using namespace tld;
using namespace cv;

enum class AppModes {
    test,
    webcam,
    videofile
};

void print_help() {
    using namespace std;
    cout << "=== TLD tracker demo application ===" << endl;
    cout << "Command line options template: '--key=val' or '--key'" << endl;
    cout << "\t--help\t outputs help reference " << endl;
    cout << "\t--mode=MODE\t runs application in appropriate mode. MODE may be set as 'test', 'webcam' or 'video'" << endl;
    cout << "\t--webcam\t same as '--mode=webcam'" << endl;
    cout << "\t--test\t same as '--mode=test'" << endl;
    cout << "\t--video\t same as '--mode=video'" << endl;
    cout << "\t--camid=ID\t selects active camera device with appropriate ID in 'webcam' mode" << endl;
    cout << "\t--videopath=PATH\t selects active video file with appropriate absolute path" << endl;
    cout << "\t--debug\t enables debug window with intermediate processsing results" << endl;
    cout << endl;
}

void run_app(VideoCapture& cap, bool debug, bool video=false) {
    if(!cap.isOpened()) {
        cout << "Error while opening cv::VideoCapture" << endl;
    }
    cv::Mat src_frame, frame, gray;
    Rect target;
    Candidate result;
    Size frame_size(640, 480);
    auto tracker = make_tld_tracker();
    size_t frame_count = std::numeric_limits<size_t>::max();
    if (video)
        frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT);
    size_t frame_id = 0;
    while(cap.isOpened()) {
        LOG_DURATION("Iteration")

        cap >> src_frame;
        resize(src_frame, frame, frame_size);
        if (frame.empty())
            break;
        cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        { LOG_DURATION("Processing")
        result = tracker << gray;
        }

        if (debug) {
            std::vector<tld::Candidate> _det_proposals;
            std::vector<tld::Candidate> _det_clusters;
            tld::Candidate _track_proposal;
            std::tie(_det_proposals, _det_clusters, _track_proposal) = tracker.GetProposals();
            auto deb_frame = frame.clone();
            tld::drawCandidates(deb_frame, _det_clusters);
            tld::drawCandidate(deb_frame, _track_proposal);
            auto psample = tracker.GetModelsPositive();
            auto nsample = tracker.GetModelsNegative();
            for (auto i = 0; i < min(5, static_cast<int>(psample.size())); i++) {
                Mat p = psample[i];
                Mat out(p.cols, p.rows, CV_8UC1);
                Mat out_color(p.cols, p.rows, CV_8UC3);
                normalize(p, out, 0, 255, NORM_MINMAX, CV_8UC1);
                cvtColor(out, out_color, cv::COLOR_GRAY2BGR);
                cv::Mat out_resized(45,45,CV_8UC3);
                resize(out_color, out_resized, Size(45, 45));
                Rect position(i*45, 0, 45, 45);
                out_resized.copyTo(deb_frame(position));
            }
            for (auto i = 0; i < min(5, static_cast<int>(nsample.size())); i++) {
                Mat n = nsample[i];
                Mat out(n.cols, n.rows, CV_8UC1);
                Mat out_color(n.cols, n.rows, CV_8UC3);
                normalize(n, out, 0, 255, NORM_MINMAX, CV_8UC1);
                cvtColor(out, out_color, cv::COLOR_GRAY2BGR);
                cv::Mat out_resized(45,45,CV_8UC3);
                resize(out_color, out_resized, Size(45, 45));
                Rect position(i*45, deb_frame.rows - 45, 45, 45);
                out_resized.copyTo(deb_frame(position));
            }
            imshow("Debug", deb_frame);
        }

        tld::drawCandidate(frame, result);
        imshow("Stream", frame);

        auto in_symbol = waitKey(1);
        if  (in_symbol == 'q')
            break;
        else if (in_symbol == 't') {
            target = selectROI("Stream", frame);
            tracker << target;
        }

        std::cout << tracker;

        frame_id++;
        if (frame_id == frame_count) {
            frame_id = 0;
            cap.set(cv::CAP_PROP_POS_FRAMES, frame_id);
        }
   }
}

int main(int argc, char** argv) {

    try {
        auto mode = AppModes::webcam;
        size_t webcam_id = 0;
        std::string video_path;
        bool debug = false;

        std::unordered_map<std::string, std::string> options = parse(argc, argv);
        for (auto [key, val]: options) {
            if (key == "help") {
                print_help();
                return 0;
            }
            if (key == "mode") {
                if (val == "webcam")
                    mode = AppModes::webcam;
                else if (val == "video")
                    mode = AppModes::videofile;
                else if (val == "test")
                    mode = AppModes::test;
            }
            if (key == "webcam")
                mode = AppModes::webcam;
            if (key == "video")
                mode = AppModes::videofile;
            if (key == "test")
                mode = AppModes::test;
            if (key == "videopath")
                video_path = val;
            if (key == "camid") {
                webcam_id = std::stoi(val);
            }
            if (key == "debug")
                debug = true;

        }

        switch (mode) {
            case AppModes::test: {
                run_tests();
            } break;
            case AppModes::videofile: {
                VideoCapture cap(video_path);
                run_app(cap, debug, true);
            } break;
            case AppModes::webcam: {
                VideoCapture cap(webcam_id);
                run_app(cap, debug);
            } break;
        }
    } catch (std::exception& e) {
        cout << "Got an exception: " << e.what() << endl;
    }

    return 0;
}
