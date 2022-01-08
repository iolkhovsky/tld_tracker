#pragma once

#include "tld_utils.h"

#include <opencv2/opencv.hpp>

namespace tld {

    class OptFlowTracker {
    public:
        OptFlowTracker();
        void SetFrame(cv::Mat frame);
        void SetTarget(cv::Rect strobe);
        Candidate Track();
    private:
        cv::Rect2d _target;
        cv::Point2d _center;
        cv::Size2d _size;
        cv::Mat _current_frame;
        cv::Mat _prev_frame;
        std::vector<cv::Point2f> _prev_points;
        std::vector<cv::Point2f> _cur_points;
        std::vector<cv::Point2f> _backtrace;
        std::vector<cv::Point2f> _prev_out_points;
        std::vector<cv::Point2f> _cur_out_points;
        std::vector<uchar> _cur_status;
        std::vector<uchar> _backtrace_status;
    };

}
