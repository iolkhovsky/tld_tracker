#include <tracker/object_detector.h>

namespace TLD {

    ObjectDetector::ObjectDetector() {
        _scales = {0.75, 0.875, 1.0, 1.125, 1.25};
        _overlap = 0.1;
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
        Augmentator aug(*_frame_ptr, _designation, aug_pars);

        _train(aug);
    }

    std::vector<Candidate> ObjectDetector::Detect() {
        return {};
    }

    void ObjectDetector::Train(Candidate prediction) {

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

    void ObjectDetector::_train(cv::Rect roi) {
        cv::Mat& frame = *_frame_ptr;
        cv::Mat subframe = frame(roi);
        for (size_t i = 0; i < _feat_extractors.size(); i++) {
             auto descriptor = _feat_extractors.at(i).GetDescriptor(subframe);
             _classifiers.at(i).TrainPositive(descriptor);
        }
    }

    void ObjectDetector::_train(Augmentator aug) {
        for (auto augm_subframe: aug.SetClass(ObjectClass::Positive)) {
            for (size_t i = 0; i < _feat_extractors.size(); i++) {
                 auto descriptor = _feat_extractors.at(i).GetDescriptor(augm_subframe);
                 _classifiers.at(i).TrainPositive(descriptor);
            }
        }
        for (auto augm_subframe: aug.SetClass(ObjectClass::Negative)) {
            for (size_t i = 0; i < _feat_extractors.size(); i++) {
                 auto descriptor = _feat_extractors.at(i).GetDescriptor(augm_subframe);
                 _classifiers.at(i).TrainNegative(descriptor);
            }
        }
    }

}
