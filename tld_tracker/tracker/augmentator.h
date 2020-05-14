#pragma once

#include <tracker/common.h>
#include <tracker/tld_utils.h>

namespace TLD {

    struct TranformPars {
        std::vector<double> scales;
        std::vector<double> angles;
        std::vector<int> translation_x;
        std::vector<int> translation_y;
        double overlap;
        double disp_threshold;
        size_t max_sample_length;
    };

    enum class ObjectClass {
        Positive,
        Negative
    };

    class Augmentator {
    public:
        Augmentator(const cv::Mat& frame, cv::Rect target, TranformPars pars);
        Augmentator& SetClass(ObjectClass name);
        void _make_positive_sample();
        void _make_negative_sample();
        double _update_target_stddev();
        std::vector<cv::Mat>::iterator begin();
        std::vector<cv::Mat>::iterator end();

    private:
        const cv::Mat& _frame;
        cv::Rect _target;
        TranformPars _pars;
        double _target_stddev;
        std::vector<cv::Mat> _sample;
    };

}
