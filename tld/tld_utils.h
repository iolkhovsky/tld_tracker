#pragma once

#include <iostream>
#include "common.h"

namespace tld {

    struct TranformPars {
        std::vector<double> scales;
        std::vector<double> angles;
        std::vector<int> translation_x;
        std::vector<int> translation_y;
        double overlap;
        double disp_threshold;
        int pos_sample_size_limit;
        int neg_sample_size_limit;
    };

    enum class ObjectClass {
        Positive,
        Negative
    };

    struct DetectorSettings {
        std::vector<double> init_training_rotation_angles = {-22.5, -20.0, -17.5, -15.0, -12.5, -10.0, -7.5, -5.0, -2.5, 0.0,
                                                        2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5};
        std::vector<double> training_rotation_angles = {-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5};
        std::vector<double> init_training_scales = {0.96, 0.98, 1.0, 1.02, 1.04};
        std::vector<double> training_scales = {1.0};
        std::vector<double> scanning_scales = {0.75, 0.875, 1.0, 1.125, 1.25};
        double scanning_overlap = 0.1;
        double training_iou_threshold = 0.3;
        double training_init_saturation = 0.4;
        double stddev_relative_threshold = 0.1;
        double detection_probability_threshold = 0.6;
        double training_pos_min_prob = 0.2;
        double training_pos_max_prob = 0.8;
        double training_neg_min_prob = 0.2;
        double training_neg_max_prob = 0.5;
    };

    struct TrackerSettings {

    };

    struct ModelSettings {

    };

    struct IntegratorSettings {
        double model_prob_threshold = 0.5;
        double tracker_prob_threshold = 0.5;
        double clusterization_iou_threshold = 0.5;
    };

    struct Settings {
        DetectorSettings detector;
        TrackerSettings tracker;
        ModelSettings model;
        IntegratorSettings integrator;
    };

    enum class ProposalSource {
        tracker,
        detector,
        mixed,
        final
    };

    struct Candidate {
        cv::Rect strobe;
        double prob;
        double aux_prob;
        bool valid;
        ProposalSource src;

        bool operator<(const Candidate& other) const;
    };

    double get_normalized_random();
    cv::Rect get_extended_rect_for_rotation(cv::Rect base_rect, double angle_degrees);
    cv::Mat get_rotated_subframe(cv::Mat frame, cv::Rect subframe_rect, double angle);
    void rotate_subframe(cv::Mat& frame, cv::Rect subframe_rect, double angle);
    uint8_t bilinear_interp_for_point(double x, double y, const cv::Mat& frame);
    double degree2rad(double angle);
    double rad2degree(double angle);
    cv::Mat subframe_linear_transform(const cv::Mat& frame, cv::Rect in_strobe, double angle,
                                        double scale, int offset_x, int offset_y);
    double compute_iou(cv::Rect a, cv::Rect b);
    cv::Point2f get_mean_shift(const std::vector<cv::Point2f> &start, const std::vector<cv::Point2f> &stop);
    double get_scale(const std::vector<cv::Point2f> &start, const std::vector<cv::Point2f> &stop);
    void drawCandidate(cv::Mat& frame, Candidate candidate);
    void drawCandidates(cv::Mat& frame, std::vector<Candidate> candidates);
    std::vector<cv::Size> get_scan_position_cnt(cv::Size frame_size, cv::Size box, std::vector<double> scales, std::vector<cv::Size> steps);
    int get_random_int(int maxint);
    double images_correlation(const cv::Mat &image_1, const cv::Mat &image_2);
    std::vector<Candidate> non_max_suppression(const std::vector<Candidate>& in, double threshold_iou);
    std::vector<tld::Candidate> clusterize_candidates(const std::vector<Candidate>& in, double threshold_iou);
    Candidate aggregate_candidates(std::vector<Candidate> sample);
    cv::Rect adjust_rect_to_frame(cv::Rect rect, cv::Size sz);
    bool strobe_is_outside(cv::Rect rect, cv::Size sz);
    cv::Mat generate_random_image();
    cv::Mat generate_random_image(cv::Size sz);
    double get_frame_std_dev(const cv::Mat& frame, cv::Rect roi);
}
