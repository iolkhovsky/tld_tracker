#include <iostream>

#include <cmdline_parser.h>
#include <tracker/tld_tracker.h>
#include <unit_tests.h>
#include <profile.h>

using namespace std;
using namespace TLD;
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
    auto tracker = make_tld_tracker();
    size_t frame_count = std::numeric_limits<size_t>::max();
    if (video)
        frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT);
    size_t frame_id = 0;
    while(cap.isOpened()) {
        LOG_DURATION("Iteration")

        cap >> src_frame;
        resize(src_frame, frame, Size(640, 480));
        if (frame.empty())
            break;
        cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        { LOG_DURATION("Processing")
        result = tracker << gray;
        }

        std::vector<TLD::Candidate> _det_proposals;
        std::vector<TLD::Candidate> _det_clusters;
        TLD::Candidate _track_proposal;
        std::tie(_det_proposals, _det_clusters, _track_proposal) = tracker.GetProposals();

        auto deb_frame = frame.clone();

        TLD::drawCandidate(frame, result);
        TLD::drawCandidates(deb_frame, _det_clusters);
        TLD::drawCandidate(deb_frame, _track_proposal);

        imshow("Stream", frame);
        if (debug)
            imshow("Debug", deb_frame);

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
