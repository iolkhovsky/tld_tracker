#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include <tracker/utils.h>

namespace TLD {

    class ObjectDetector {
    public:
        void SetFrame(cv::Mat& img);
        void SetTarget(cv::Rect strobe);
        void Train(Candidate prediction);
        std::vector<Candidate> Detect();
    };

}
