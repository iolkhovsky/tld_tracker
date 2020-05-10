#pragma once

#include <opencv2/opencv.hpp>

#include <tracker/scanning_grid.h>

namespace TLD {

    using BinaryDescriptor = uint16_t;

    class FeatureExtractor {
    public:
        BinaryDescriptor GetDescriptor(cv::Mat& frame, std::vector<PixelIdPair> pairs);
        BinaryDescriptor operator()(cv::Mat& frame, std::vector<PixelIdPair> pairs);
    };

}
