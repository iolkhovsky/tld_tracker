#include <tracker/object_detector.h>

namespace tld {

    void ObjectDetector::SetFrame(std::shared_ptr<cv::Mat> img) {
        _frame_ptr = img;
        _frame_size.width = img->cols;
        _frame_size.height = img->rows;
    }

    void ObjectDetector::Config(DetectorSettings settings) {
        _settings = settings;
    }

    void ObjectDetector::SetTarget(cv::Rect strobe) {
        _designation = strobe;
        _designation_stddev = get_frame_std_dev(*_frame_ptr, _designation);
        _reset();

        TranformPars aug_pars;
        aug_pars.pos_sample_size_limit = -1;
        aug_pars.neg_sample_size_limit = -1;
        aug_pars.disp_threshold = _settings.stddev_relative_threshold;
        aug_pars.angles = _settings.init_training_rotation_angles;
        aug_pars.scales = _settings.init_training_scales;
        aug_pars.overlap = _settings.scanning_overlap;
        aug_pars.translation_x = {static_cast<int>(-0.5 * _settings.scanning_overlap * _designation.width), 0,
                                  static_cast<int>(0.5 * _settings.scanning_overlap * _designation.width)};
        aug_pars.translation_y = {static_cast<int>(-0.5 * _settings.scanning_overlap * _designation.height), 0,
                                  static_cast<int>(0.5 * _settings.scanning_overlap * _designation.height)};

        Augmentator aug(*_frame_ptr, _designation, aug_pars);

        _init_train(aug);
    }

    std::vector<Candidate> ObjectDetector::Detect() {
        std::vector<Candidate> out;
        std::vector<cv::Size> positions_per_scale = _scanning_grids.front()->GetPositionsCnt();
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
                    if (ensemble_prob > _settings.detection_probability_threshold) {
                        Candidate candidate;
                        candidate.src = ProposalSource::detector;
                        candidate.prob = ensemble_prob;
                        candidate.strobe.x = x_i * _scanning_grids.front()->GetSteps()[scale_id].width;
                        candidate.strobe.y = y_i * _scanning_grids.front()->GetSteps()[scale_id].height;
                        candidate.strobe.width = _scanning_grids.front()->GetBBoxSizes()[scale_id].width;
                        candidate.strobe.height = _scanning_grids.front()->GetBBoxSizes()[scale_id].height;
                        //double proposal_disp = get_frame_std_dev(*_frame_ptr, candidate.strobe);
                        //if (proposal_disp > 0.1 * _designation_stddev)
                        out.push_back(candidate);
                    }
                }
            }
            scale_id++;
        }

        std::sort(out.begin(), out.end(), [] (const Candidate& lhs, const Candidate& rhs) {
           return lhs.prob > rhs.prob;
        });

        return {out.begin(), std::min(out.end(), std::next(out.begin(), 16))};
    }

    void ObjectDetector::Train(Candidate prediction) {
        TranformPars aug_pars;
        aug_pars.angles = _settings.training_rotation_angles;
        aug_pars.scales = _settings.training_scales;
        aug_pars.translation_x = {0};
        aug_pars.translation_x = {0};
        aug_pars.overlap = _settings.scanning_overlap;
        aug_pars.disp_threshold = _settings.stddev_relative_threshold;
        aug_pars.pos_sample_size_limit = -1;
        aug_pars.neg_sample_size_limit = -1;
        Augmentator aug(*_frame_ptr, prediction.strobe, aug_pars);
        _train(aug);
    }

    void ObjectDetector::_reset() {
        _feat_extractors.clear();
        _scanning_grids.clear();
        _classifiers.clear();
        for (auto i = 0; i < CLASSIFIERS_CNT; i++) {
            _scanning_grids.push_back(std::make_shared<ScanningGrid>(_frame_size));
            _scanning_grids.back()->SetBase({_designation.width, _designation.height}, 0.1, _settings.scanning_scales);
            _feat_extractors.push_back(_scanning_grids.back());
            _classifiers.emplace_back(ObjectClassifier<BinaryDescriptor, BINARY_DESCRIPTOR_CNT>());
        }
    }

    void ObjectDetector::UpdateGrid(const Candidate& reference) {
        _designation = reference.strobe;
        for (auto& grid: _scanning_grids)
            grid->SetBase({_designation.width, _designation.height}, _settings.scanning_overlap, _settings.scanning_scales);
    }

    void ObjectDetector::_train(Augmentator aug) {
        for (auto augm_subframe: aug.SetClass(ObjectClass::Positive)) {
            double ensemble_prob = _ensamble_prediction(augm_subframe);
            if ((ensemble_prob >= _settings.training_pos_min_prob)
                    && (ensemble_prob <= _settings.training_pos_max_prob)) {
                for (size_t i = 0; i < _feat_extractors.size(); i++) {
                     auto descriptor = _feat_extractors.at(i).GetDescriptor(augm_subframe);
                     _classifiers.at(i).TrainPositive(descriptor);
                }
            }
        }
        for (auto augm_subframe: aug.SetClass(ObjectClass::Negative)) {
            double ensemble_prob = _ensamble_prediction(augm_subframe);
            if ((ensemble_prob >= _settings.training_neg_min_prob)
                    && (ensemble_prob <= _settings.training_neg_max_prob)) {
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
                if (_classifiers.at(i).GetNegativeDistr(descriptor) < static_cast<size_t>(pos_max * _settings.training_init_saturation))
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
