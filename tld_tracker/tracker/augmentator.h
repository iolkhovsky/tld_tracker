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
    };

    enum class ObjectClass {
        Positive,
        Negative
    };

    class Augmentator {
    public:
        Augmentator(const cv::Mat& frame, cv::Rect target, TranformPars pars) :
            _frame(frame), _target(target), _pars(pars) {
        }

        Augmentator& SetClass(ObjectClass name) {

            if (name == ObjectClass::Positive) {
                _make_positive_sample();
            } else if (name == ObjectClass::Negative) {
                _make_negative_sample();
            } else
                throw std::runtime_error("Unexpected object class in Augmentator");

            return *this;
        }

        void _make_positive_sample() {
            _sample.clear();

            for (auto angle: _pars.angles)
                for (auto scale: _pars.scales)
                    for (auto transl_x: _pars.translation_x)
                        for (auto transl_y: _pars.translation_y)
                            _sample.push_back(subframe_linear_transform(_frame, _target, angle, scale,
                                                                        transl_x, transl_y));

        }

        void _make_negative_sample() {
            _sample.clear();


        }

        std::vector<cv::Mat>::iterator begin() {
            return std::begin(_sample);
        }

        std::vector<cv::Mat>::iterator end() {
            return std::end(_sample);
        }

    private:
        const cv::Mat& _frame;
        cv::Rect _target;
        TranformPars _pars;
        std::vector<cv::Mat> _sample;
    };

}
