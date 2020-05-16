#pragma once

#include <inttypes.h>
#include <utility>
#include <opencv2/opencv.hpp>
#include <algorithm>

#define BINARY_DESCRIPTOR_WIDTH     (10)
#define BINARY_DESCRIPTOR_CNT       (1024)
#define CLASSIFIERS_CNT             (10)

namespace TLD {
    using BinaryDescriptor = uint16_t;
    using NormFernPair = std::pair<cv::Point2d, cv::Point2d>;
    using AbsFernPair = std::pair<cv::Point2i, cv::Point2i>;
    using PixelIdPair = std::pair<size_t, size_t>;
}
