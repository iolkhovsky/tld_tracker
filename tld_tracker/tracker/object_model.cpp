#include <tracker/object_model.h>

namespace TLD {

    ObjectModel::ObjectModel() {
        _sample_max_depth = 100;
        _scales = {0.75, 0.875, 1.0, 1.125, 1.25};
        _overlap = 0.1;
    }

    void ObjectModel::SetFrame(std::shared_ptr<cv::Mat> frame) {
        _frame = frame;
    }

    void ObjectModel::SetTarget(cv::Rect target) {
        _target = target;
    }

    void ObjectModel::Train(Candidate prediction) {
        const cv::Mat& src_frame = *_frame;

        TranformPars aug_pars;
        aug_pars.angles = {-15, 0, 15};
        aug_pars.scales = _scales;
        aug_pars.translation_x = {static_cast<int>(-0.5*_overlap * prediction.strobe.width), 0,
                                  static_cast<int>(0.5*_overlap * prediction.strobe.width)};
        aug_pars.translation_x = {static_cast<int>(-0.5*_overlap * prediction.strobe.height), 0,
                                  static_cast<int>(0.5*_overlap * prediction.strobe.height)};
        aug_pars.overlap = _overlap;
        Augmentator aug(src_frame, prediction.strobe, aug_pars);

        for (auto subframe: aug.SetClass(ObjectClass::Positive)) {
            cv::Mat patch;
            cv::resize(subframe, patch, _patch_size);
            _positive_sample.push_back(std::move(patch));
        }

        for (auto subframe: aug.SetClass(ObjectClass::Negative)) {
            cv::Mat patch;
            cv::resize(subframe, patch, _patch_size);
            _negative_sample.push_back(std::move(patch));
        }

    }

    double ObjectModel::Predict(Candidate candidate) {

    }

}
