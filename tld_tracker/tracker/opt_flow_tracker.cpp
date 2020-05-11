#include <tracker/opt_flow_tracker.h>

namespace TLD {

    OptFlowTracker::OptFlowTracker() {

    }

    void OptFlowTracker::SetFrame(cv::Mat frame) {
        _prev_frame = std::move(_current_frame);
        _current_frame = std::move(frame);
    }

    void OptFlowTracker::SetTarget(cv::Rect strobe) {
        _target = strobe;
    }

    Candidate OptFlowTracker::Track() {
        if ((_target.area() == 0) || _prev_frame.empty() || _current_frame.empty())
            return {};
        cv::Mat mask = cv::Mat(_prev_frame.rows, _prev_frame.cols, CV_8UC1);
        cv::Mat target_subframe = mask(_target);
        target_subframe.setTo(255);
        cv::goodFeaturesToTrack(_prev_frame, _prev_points, 100, 0.3, 7, mask, 7, false, 0.04);
        // calculate optical flow
        std::vector<uint8_t> status;
        std::vector<float> err;
        cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
        cv::calcOpticalFlowPyrLK(_prev_frame, _current_frame, _prev_points, _cur_points, status, err, cv::Size(15,15), 2, criteria);
    }

}


