#pragma once

#include <iostream>
#include <tracker/common.h>

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
    uint8_t bilinear_interp_for_point(double x, double y, const cv::Mat& frame);
    double degree2rad(double angle);
    double rad2degree(double angle);
    cv::Mat subframe_linear_transform(const cv::Mat& frame,
                                        cv::Rect strobe,
                                        double angle,
                                        double scale,
                                        int offset_x,
                                        int offset_y);
    double iou(cv::Rect a, cv::Rect b);
    cv::Point2f get_mean_shift(const std::vector<cv::Point2f> &start, const std::vector<cv::Point2f> &stop);
    void drawCandidate(cv::Mat& frame, Candidate candidate);

}
