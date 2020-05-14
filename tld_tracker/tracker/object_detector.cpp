#include <tracker/object_detector.h>

namespace TLD {

    ObjectDetector::ObjectDetector() {
        _scales = {0.75, 0.875, 1.0, 1.125, 1.25};
        _overlap = 0.1;
        _saturation_limit = 0.4;
    }

    void ObjectDetector::SetFrame(std::shared_ptr<cv::Mat> img) {
        _frame_ptr = img;
        _frame_size.width = img->cols;
        _frame_size.height = img->rows;
    }

    void ObjectDetector::SetTarget(cv::Rect strobe) {
        _designation = strobe;
        _reset();

        TranformPars aug_pars;
        aug_pars.angles = {-15, 0, 15};
        aug_pars.scales = _scales;
        aug_pars.translation_x = {static_cast<int>(-0.5*_overlap * _designation.width), 0,
                                  static_cast<int>(0.5*_overlap * _designation.width)};
        aug_pars.translation_x = {static_cast<int>(-0.5*_overlap * _designation.height), 0,
                                  static_cast<int>(0.5*_overlap * _designation.height)};
        aug_pars.overlap = _overlap;
        aug_pars.disp_threshold = 0.5;
        aug_pars.max_sample_length = 100;

        Augmentator aug(*_frame_ptr, _designation, aug_pars);

        _init_train(aug);
    }

    std::vector<Candidate> ObjectDetector::Detect() {
        std::vector<Candidate> out;
        std::vector<cv::Size> positions_per_scale = _scanning_grids.at(0).GetPositionsCnt();
        size_t scale_id = 0;
        for (auto positions: positions_per_scale) {
            for (auto y_i = 0; y_i < positions.height; y_i++) {
                for (auto x_i = 0; x_i < positions.width; x_i++) {
                    double ensemble_prob = 0.0;
                    for (size_t i = 0; i < _classifiers.size(); i++) {
                        auto desc = _feat_extractors.at(i)(*_frame_ptr, {x_i, y_i}, scale_id);
                        ensemble_prob += _classifiers.at(i).Predict(desc);
                    }
                    ensemble_prob /= _classifiers.size();
                    if (ensemble_prob > 0.5) {
                        Candidate candidate;
                        candidate.src = ProposalSource::detector;
                        candidate.prob = ensemble_prob;
                        candidate.strobe.x = x_i * _scanning_grids.at(0).GetOverlap().width;
                        candidate.strobe.y = y_i * _scanning_grids.at(0).GetOverlap().width;
                        candidate.strobe.width = static_cast<int>(_scales.at(scale_id) * _designation.width);
                        candidate.strobe.height = static_cast<int>(_scales.at(scale_id) * _designation.height);
                        out.push_back(candidate);
                    }
                }
            }
            scale_id++;
        }

        return out;
    }

    void ObjectDetector::Train(Candidate prediction) {
        TranformPars aug_pars;
        aug_pars.angles = {-15, 0, 15};
        aug_pars.scales = _scales;
        aug_pars.translation_x = {0};
        aug_pars.translation_x = {0};
        aug_pars.overlap = _overlap;
        aug_pars.disp_threshold = 0.5;
        aug_pars.max_sample_length = 25;
        Augmentator aug(*_frame_ptr, prediction.strobe, aug_pars);
        _train(aug);
    }

    void ObjectDetector::_reset() {
        _feat_extractors.clear();
        _scanning_grids.clear();
        _classifiers.clear();
        for (auto i = 0; i < CLASSIFIERS_CNT; i++) {
            _scanning_grids.emplace_back(ScanningGrid(_frame_size));
            _scanning_grids.back().SetBase({_designation.width, _designation.height}, 0.1, _scales);
            _feat_extractors.emplace_back(FernFeatureExtractor(_scanning_grids.back()));
            _classifiers.emplace_back(ObjectClassifier<BinaryDescriptor, BINARY_DESCRIPTOR_CNT>());
        }
    }

    void ObjectDetector::_train(Augmentator aug) {
        for (auto augm_subframe: aug.SetClass(ObjectClass::Positive)) {
            double ensemble_prob = _ensamble_prediction(augm_subframe);
            if ((ensemble_prob >= 0.2) && (ensemble_prob <= 0.8)) {
                for (size_t i = 0; i < _feat_extractors.size(); i++) {
                     auto descriptor = _feat_extractors.at(i).GetDescriptor(augm_subframe);
                     _classifiers.at(i).TrainPositive(descriptor);
                }
            }
        }
        for (auto augm_subframe: aug.SetClass(ObjectClass::Negative)) {
            double ensemble_prob = _ensamble_prediction(augm_subframe);
            if ((ensemble_prob >= 0.2) && (ensemble_prob <= 0.5)) {
                for (size_t i = 0; i < _feat_extractors.size(); i++) {
                    auto descriptor = _feat_extractors.at(i).GetDescriptor(augm_subframe);
                    _classifiers.at(i).TrainNegative(descriptor);
                }
            }
        }
    }

    void ObjectDetector::_init_train(Augmentator aug) {
        for (auto augm_subframe: aug.SetClass(ObjectClass::Positive)) {
            for (size_t i = 0; i < _feat_extractors.size(); i++) {
                 auto descriptor = _feat_extractors.at(i).GetDescriptor(augm_subframe);
                 _classifiers.at(i).TrainPositive(descriptor);
            }
        }
        for (auto augm_subframe: aug.SetClass(ObjectClass::Negative)) {
            for (size_t i = 0; i < _feat_extractors.size(); i++) {
                auto descriptor = _feat_extractors.at(i).GetDescriptor(augm_subframe);
                size_t pos_max = _classifiers.at(i).GetMaxPositive();
                if (_classifiers.at(i).GetNegativeDistr(descriptor) < static_cast<size_t>(pos_max * _saturation_limit))
                    _classifiers.at(i).TrainNegative(descriptor);
            }
        }
    }

    double ObjectDetector::_ensamble_prediction(cv::Mat img) {
        double accum = 0.0;
        for (size_t i = 0; i < _feat_extractors.size(); i++) {
            auto descriptor = _feat_extractors.at(i).GetDescriptor(img);
            accum += _classifiers.at(i).Predict(descriptor);
        }
        accum /= _feat_extractors.size();
        return accum;
    }

}
