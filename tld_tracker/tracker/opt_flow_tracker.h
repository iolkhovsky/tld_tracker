#pragma once

#include <opencv2/opencv.hpp>
#include <tracker/tld_utils.h>

namespace TLD {

    class OptFlowTracker {
    public:
        OptFlowTracker();
        void SetFrame(cv::Mat frame);
        void SetTarget(cv::Rect strobe);
        Candidate Track();
    private:
        cv::Rect _target;
        cv::Mat _current_frame;
        cv::Mat _prev_frame;
        std::vector<cv::Point2f> _prev_points, _cur_points;
    };

}
