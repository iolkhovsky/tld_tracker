#pragma once

#include <opencv2/opencv.hpp>
#include <tracker/utils.h>

namespace TLD {

    class OptFlowTracker {
    public:
        OptFlowTracker();
        void SetFrame(cv::Mat& frame);
        void SetTarget(cv::Rect strobe);
        Candidate Track();
    private:
        cv::Mat _current_frame;
        cv::Mat _prev_frame;
    };

}
