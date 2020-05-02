#pragma once

#include <opencv2/opencv.hpp>
#include <tracker/utils.h>

namespace TLD {

    class ObjectModel {
    public:
        ObjectModel();
        void SetFrame(cv::Mat& frame);
        void SetTarget(cv::Rect target);
        void Train(Candidate prediction);
    };

}
