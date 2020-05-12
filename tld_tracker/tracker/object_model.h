#pragma once

#include <tracker/tld_utils.h>
#include <tracker/augmentator.h>

namespace TLD {

    class ObjectModel {
    public:
        ObjectModel();
        void SetFrame(std::shared_ptr<cv::Mat> frame);
        void SetTarget(cv::Rect target);
        void Train(Candidate prediction);
        double Predict(Candidate candidate);
    private:
        cv::Size _patch_size;
        std::shared_ptr<cv::Mat> _frame;
        cv::Rect _target;
        std::vector<double> _scales;
        double _overlap;

        size_t _sample_max_depth;
        std::vector<cv::Mat> _positive_sample;
        std::vector<cv::Mat> _negative_sample;
    };

}
