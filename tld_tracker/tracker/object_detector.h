#pragma once

#include <vector>
#include <memory>

#include <tracker/common.h>
#include <tracker/tld_utils.h>
#include <tracker/fern_fext.h>
#include <tracker/object_classifier.h>
#include <tracker/augmentator.h>

namespace TLD {

    class ObjectDetector {
    public:
        ObjectDetector();
        void SetFrame(std::shared_ptr<cv::Mat> img);
        void SetTarget(cv::Rect strobe);
        void Train(Candidate prediction);
        std::vector<Candidate> Detect();
        double _ensamble_prediction(cv::Mat img);
    private:
        std::shared_ptr<cv::Mat> _frame_ptr;
        cv::Size _frame_size;
        std::vector<ScanningGrid> _scanning_grids;
        std::vector<FernFeatureExtractor> _feat_extractors;
        std::vector<ObjectClassifier<BinaryDescriptor, BINARY_DESCRIPTOR_CNT>> _classifiers;
        cv::Rect _designation;
        std::vector<double> _scales;
        double _overlap;
        double _saturation_limit;

        void _reset();
        void _train(Augmentator aug);
        void _init_train(Augmentator aug);
    };

}
