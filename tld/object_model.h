#pragma once

#include "tld_utils.h"
#include "augmentator.h"

namespace tld {

    class ObjectModel {
    public:
        ObjectModel();
        void SetFrame(std::shared_ptr<cv::Mat> frame);
        void SetTarget(cv::Rect target);
        void Train(Candidate candidate);
        double Predict(Candidate candidate) const;
        double Predict(const cv::Mat& subframe) const;
        std::vector<cv::Mat> GetPositiveSample() const;
        std::vector<cv::Mat> GetNegativeSample() const;
    private:
        cv::Size _patch_size;
        std::shared_ptr<cv::Mat> _frame;
        cv::Rect _target;
        std::vector<double> _scales;
        double _init_overlap;
        double _overlap;

        size_t _sample_max_depth;
        std::vector<cv::Mat> _positive_sample;
        std::vector<cv::Mat> _negative_sample;

        cv::Mat _make_patch(const cv::Mat& subframe) const;
        void _add_new_patch(cv::Mat&& patch, std::vector<cv::Mat>& sample);
        double _similarity_coeff(const cv::Mat& subframe_0, cv::Mat& subframe_1) const;
        double _sample_dissimilarity(cv::Mat& patch, const std::vector<cv::Mat>& sample) const;
        double _predict(cv::Mat& patch) const;
    };

}
