#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdlib>

std::ostream& operator<<(std::ostream &os, const cv::Rect& rect);

namespace TLD {

    enum class ProposalSource {
        tracker,
        detector
    };

    struct Candidate {
        cv::Rect strobe;
        double prob;
        bool valid;
        bool training;
        ProposalSource src;
    };

    double get_normalized_random();
    cv::Rect get_extended_rect_for_rotation(cv::Rect base_rect, double angle_degrees);
    cv::Mat get_rotated_subframe(cv::Mat frame, cv::Rect subframe_rect, double angle);
    void rotate_subframe(cv::Mat& frame, cv::Rect subframe_rect, double angle);

}
