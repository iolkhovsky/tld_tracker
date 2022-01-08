#pragma once

#include <vector>
#include <memory>

#include "common.h"
#include "tld_utils.h"
#include "fern_fext.h"
#include "object_classifier.h"
#include "augmentator.h"

namespace tld {

    class ObjectDetector {
    public:
        ObjectDetector() = default;
        void SetFrame(std::shared_ptr<cv::Mat> img);
        void SetTarget(cv::Rect strobe);
        void UpdateGrid(const Candidate& reference);
        void Train(Candidate prediction);
        std::vector<Candidate> Detect();
        double _ensamble_prediction(cv::Mat img);
        void Config(DetectorSettings settings);
    private:
        std::shared_ptr<cv::Mat> _frame_ptr;
        cv::Size _frame_size;
        std::vector<std::shared_ptr<ScanningGrid>> _scanning_grids;
        std::vector<FernFeatureExtractor> _feat_extractors;
        std::vector<ObjectClassifier<BinaryDescriptor, BINARY_DESCRIPTOR_CNT>> _classifiers;
        cv::Rect _designation;
        double _designation_stddev;
        DetectorSettings _settings;

        void _reset();
        void _train(Augmentator aug);
        void _init_train(Augmentator aug);
    };

}
