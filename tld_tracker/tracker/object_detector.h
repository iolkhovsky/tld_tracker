#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

#include <tracker/utils.h>

namespace TLD {

    class ObjectDetector {
    public:
        ObjectDetector();
        void SetFrame(std::shared_ptr<cv::Mat> img);
        void SetTarget(cv::Rect strobe);
        void Train(Candidate prediction);
        std::vector<Candidate> Detect();
    private:
        std::shared_ptr<cv::Mat> _frame_ptr;
    };

}
