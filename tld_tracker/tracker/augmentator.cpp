#include <tracker/augmentator.h>

namespace tld {

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

        int samples_count = 0;
        for (auto angle: _pars.angles) {
            for (auto scale: _pars.scales) {
                for (auto transl_x: _pars.translation_x) {
                    for (auto transl_y: _pars.translation_y) {
                        _sample.push_back(subframe_linear_transform(_frame, _target, angle, scale,
                                                                    transl_x, transl_y));
                        samples_count++;
                        if ((samples_count >= _pars.pos_sample_size_limit) && (_pars.pos_sample_size_limit > 0))
                            break;
                    }
                    if ((samples_count >= _pars.pos_sample_size_limit) && (_pars.pos_sample_size_limit > 0))
                        break;
                }
                if ((samples_count >= _pars.pos_sample_size_limit) && (_pars.pos_sample_size_limit > 0))
                    break;
            }
            if ((samples_count >= _pars.pos_sample_size_limit) && (_pars.pos_sample_size_limit > 0))
                break;
        }
    }

    void Augmentator::_make_negative_sample() {
        _sample.clear();
        auto target_stddev = _update_target_stddev();
        std::vector<cv::Size> steps;
        for (auto scale: _pars.scales) {
            cv::Size scaled_bbox(static_cast<int>(_target.width * scale),
                                 static_cast<int>(_target.height * scale));
            int step_x = static_cast<int>(scaled_bbox.width * _pars.overlap);
            int step_y = static_cast<int>(scaled_bbox.height * _pars.overlap);
            step_x = std::max(4, step_x);
            step_y = std::max(4, step_y);
            steps.push_back({step_x, step_y});
        }

        auto scan_positions = get_scan_position_cnt(_frame.size(), {_target.width, _target.height},
                                                    _pars.scales, steps);
        int samples_count = 0;
        for (size_t scale_id = 0; scale_id < _pars.scales.size(); scale_id++) {
            cv::Rect current_rect = {0, 0, static_cast<int>(_target.width * _pars.scales.at(scale_id)),
                                     static_cast<int>(_target.height * _pars.scales.at(scale_id))};
            int step_x = steps.at(scale_id).width;
            int step_y = steps.at(scale_id).height;
            cv::Size positions = scan_positions.at(scale_id);
            for (auto x_org = 0; x_org < positions.width; x_org ++) {
                for (auto y_org = 0; y_org < positions.height; y_org ++) {
                    current_rect.x = x_org * step_x;
                    current_rect.y = y_org * step_y;
                    double iou = compute_iou(current_rect, _target);
                    double stddev = get_frame_std_dev(_frame, current_rect);
                    if ((iou < 0.1) && (stddev > target_stddev * _pars.disp_threshold)) {
                        _sample.push_back(_frame(current_rect).clone());
                        samples_count++;
                    }
                    if ((samples_count >= _pars.neg_sample_size_limit) && (_pars.neg_sample_size_limit > 0))
                        break;
                }
                if ((samples_count >= _pars.neg_sample_size_limit) && (_pars.neg_sample_size_limit > 0))
                    break;
            }
            if ((samples_count >= _pars.neg_sample_size_limit) && (_pars.neg_sample_size_limit > 0))
                break;
        }
    }

    double Augmentator::_update_target_stddev() {
        cv::Mat variance, mean;
        auto target_outside_frame = strobe_is_outside(_target, {_frame.cols, _frame.rows});
        if (target_outside_frame)
            return _target_stddev;
        _target_stddev =  get_frame_std_dev(_frame, _target);
        return _target_stddev;
    }

    std::vector<cv::Mat>::iterator Augmentator::begin() {
        return std::begin(_sample);
    }

    std::vector<cv::Mat>::iterator Augmentator::end() {
        return std::end(_sample);
    }

}
