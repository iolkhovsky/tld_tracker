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
    double compute_iou(cv::Rect a, cv::Rect b);
    cv::Point2f get_mean_shift(const std::vector<cv::Point2f> &start, const std::vector<cv::Point2f> &stop);
    void drawCandidate(cv::Mat& frame, Candidate candidate);
    void drawCandidates(cv::Mat& frame, std::vector<Candidate> candidates);
    std::vector<cv::Size> get_scan_position_cnt(cv::Size frame_size, cv::Size box, std::vector<double> scales, double overlap);
    int get_random_int(int maxint);
    double images_correlation(cv::Mat &image_1, cv::Mat &image_2);

}
