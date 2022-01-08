#include "object_model.h"

namespace tld {

    ObjectModel::ObjectModel() :
        _patch_size(15,15),
        _scales({1.0}),
        _init_overlap(0.5),
        _overlap(1.0),
        _sample_max_depth(100) {
    }

    void ObjectModel::SetFrame(std::shared_ptr<cv::Mat> frame) {
        _frame = frame;
    }

    void ObjectModel::SetTarget(cv::Rect target) {
        _target = target;

        const cv::Mat& frame = *_frame;

        TranformPars aug_pars;
        aug_pars.angles = {-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90};
        aug_pars.scales = _scales;
        aug_pars.translation_x = {static_cast<int>(-0.5*_init_overlap * _target.width), 0,
                                  static_cast<int>(0.5*_init_overlap * _target.width)};
        aug_pars.translation_y = {static_cast<int>(-0.5*_init_overlap * _target.height), 0,
                                  static_cast<int>(0.5*_init_overlap * _target.height)};
        aug_pars.overlap = _init_overlap;
        aug_pars.disp_threshold = 0.1;
        aug_pars.pos_sample_size_limit = 100;
        aug_pars.neg_sample_size_limit = 100;
        Augmentator aug(frame, _target, aug_pars);

        for (auto subframe: aug.SetClass(ObjectClass::Positive)) {
            cv::Mat patch = _make_patch(subframe);
            _add_new_patch(std::move(patch), _positive_sample);
        }

        for (auto subframe: aug.SetClass(ObjectClass::Negative)) {
            cv::Mat patch = _make_patch(subframe);
            _add_new_patch(std::move(patch), _negative_sample);
        }
    }

    void ObjectModel::Train(Candidate candidate) {
        const cv::Mat& frame = *_frame;

        TranformPars aug_pars;
        aug_pars.angles = {-15, 0, 15};
        aug_pars.scales = _scales;
        aug_pars.translation_x = {static_cast<int>(-0.5*_init_overlap * _target.width), 0,
                                  static_cast<int>(0.5*_init_overlap * _target.width)};
        aug_pars.translation_y = {static_cast<int>(-0.5*_init_overlap * _target.height), 0,
                                  static_cast<int>(0.5*_init_overlap * _target.height)};
        aug_pars.neg_sample_size_limit = 24;
        aug_pars.overlap = _overlap;
        aug_pars.disp_threshold = 0.25;
        aug_pars.pos_sample_size_limit = 100;
        Augmentator aug(frame, candidate.strobe, aug_pars);

        for (auto subframe: aug.SetClass(ObjectClass::Positive)) {
            cv::Mat patch = _make_patch(subframe);
            double prob = _predict(patch);
            if (prob < 0.9)
                _add_new_patch(std::move(patch), _positive_sample);
        }

        for (auto subframe: aug.SetClass(ObjectClass::Negative)) {
            cv::Mat patch = _make_patch(subframe);
            double prob = _predict(patch);
            if (prob > 0.1)
                _add_new_patch(std::move(patch), _negative_sample);
        }
    }

    double ObjectModel::Predict(Candidate candidate) const {
        const cv::Mat& src_frame = *_frame;
        candidate.strobe = adjust_rect_to_frame(candidate.strobe, {src_frame.cols, src_frame.rows});

        cv::Mat subframe = src_frame(candidate.strobe);
        auto patch = _make_patch(subframe);
        if (patch.empty())
            return 0.0;
        else
            return _predict(patch);
    }

    double ObjectModel::Predict(const cv::Mat& subframe) const {
        auto patch = _make_patch(subframe);
        if (patch.empty())
            return 0.0;
        else
            return _predict(patch);
    }

    void ObjectModel::_add_new_patch(cv::Mat&& patch, std::vector<cv::Mat>& sample) {
        if (_sample_max_depth) {
            if (sample.size() < _sample_max_depth)
                sample.push_back(std::move(patch));
            else {
                size_t idx_2_replace = static_cast<size_t>(get_random_int(static_cast<int>(sample.size() - 1)));
                sample.at(idx_2_replace) = std::move(patch);
            }
        }
    }

    cv::Mat ObjectModel::_make_patch(const cv::Mat& subframe) const {
        if (!subframe.empty()) {
            cv::Mat patch;
            cv::resize(subframe, patch, _patch_size);
            return patch;
        } else
            return {};
    }

    double ObjectModel::_similarity_coeff(const cv::Mat& patch_0, cv::Mat& patch_1) const {
        double ncc = images_correlation(patch_0, patch_1);
        double res = 0.5 * (ncc + 1);
        return res;
    }

    double ObjectModel::_sample_dissimilarity(cv::Mat& patch, const std::vector<cv::Mat>& sample) const {
        double out = 0.0;
        for (const auto& sample_patch: sample) {
            out = std::max(_similarity_coeff(sample_patch, patch), out);
        }
        return 1 - out;
    }

    double ObjectModel::_predict(cv::Mat& patch) const {
        double out = 0.0;
        double Npm = _sample_dissimilarity(patch, _negative_sample);
        double Ppm = _sample_dissimilarity(patch, _positive_sample);
        if (abs(Npm + Ppm)>1e-9)
            out = Npm / (Npm + Ppm);
        return out;
    }

    std::vector<cv::Mat> ObjectModel::GetPositiveSample() const {
        return _positive_sample;
    }

    std::vector<cv::Mat> ObjectModel::GetNegativeSample() const {
        return _negative_sample;
    }

}
