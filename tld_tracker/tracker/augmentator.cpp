#include <tracker/augmentator.h>

namespace TLD {

    Augmentator::Augmentator(const cv::Mat& frame, cv::Rect target, TranformPars pars) :
        _frame(frame), _target(target), _pars(pars) {
    }

    Augmentator& Augmentator::SetClass(ObjectClass name) {

        if (name == ObjectClass::Positive) {
            _make_positive_sample();
        } else if (name == ObjectClass::Negative) {
            _make_negative_sample();
        } else
            throw std::runtime_error("Unexpected object class in Augmentator");

        return *this;
    }

    void Augmentator::_make_positive_sample() {
        _sample.clear();

        size_t samples_count = 0;
        for (auto angle: _pars.angles) {
            for (auto scale: _pars.scales) {
                for (auto transl_x: _pars.translation_x) {
                    for (auto transl_y: _pars.translation_y) {
                        _sample.push_back(subframe_linear_transform(_frame, _target, angle, scale,
                                                                    transl_x, transl_y));
                        samples_count++;
                        if (samples_count >= _pars.max_sample_length)
                            break;
                    }
                    if (samples_count >= _pars.max_sample_length)
                        break;
                }
                if (samples_count >= _pars.max_sample_length)
                    break;
            }
            if (samples_count >= _pars.max_sample_length)
                break;
        }
    }

    void Augmentator::_make_negative_sample() {
        _sample.clear();
        auto target_stddev = _update_target_stddev();
        auto scan_positions = get_scan_position_cnt(_frame.size(), {_target.width, _target.height},
                                                    _pars.scales, _pars.overlap);
        size_t samples_count = 0;
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
                    if ((iou < 0.1) && (stddev > target_stddev * _pars.disp_threshold)) {
                        _sample.push_back(_frame(current_rect).clone());
                        samples_count++;
                    }
                    if (samples_count >= _pars.max_sample_length)
                        break;
                }
                if (samples_count >= _pars.max_sample_length)
                    break;
            }
            if (samples_count >= _pars.max_sample_length)
                break;
        }
    }

    double Augmentator::_update_target_stddev() {
        cv::Mat variance, mean;
        cv::meanStdDev(_frame(_target), mean, variance);
        _target_stddev = variance.at<double>(0,0);
        return _target_stddev;
    }

    std::vector<cv::Mat>::iterator Augmentator::begin() {
        return std::begin(_sample);
    }

    std::vector<cv::Mat>::iterator Augmentator::end() {
        return std::end(_sample);
    }

}
