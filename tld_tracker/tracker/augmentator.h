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

            auto scan_positions = get_scan_position_cnt(_frame.size(), {_target.width, _target.height},
                                                        _pars.scales, _pars.overlap);
            for (size_t scale_id = 0; scale_id < _pars.scales.size(); scale_id++) {
                cv::Rect current_rect = {0, 0, static_cast<int>(_target.width * _pars.scales.at(scale_id)),
                                         static_cast<int>(_target.height * _pars.scales.at(scale_id))};
                int step_x = static_cast<int>(_pars.overlap * current_rect.width);
                int step_y = static_cast<int>(_pars.overlap * current_rect.height);
                cv::Size positions = scan_positions.at(scale_id);
                for (auto x_org = 0; x_org < positions.width; x_org += step_x) {
                    for (auto y_org = 0; y_org < positions.height; y_org += step_y) {
                        current_rect.x = x_org;
                        current_rect.y = y_org;
                        double iou = compute_iou(current_rect, _target);
                        cv::Mat variance, mean;
                        cv::meanStdDev(_frame(current_rect), mean, variance);
                        double stddev = variance.at<double>(0,0);
                        double avg = mean.at<double>(0,0);
                        if ((iou < 0.1) && (stddev > 10))
                            _sample.push_back(_frame(current_rect).clone());
                    }
                }

            }

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
