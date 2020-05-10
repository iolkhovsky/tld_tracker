#pragma once

#include <opencv2/opencv.hpp>

#include <tracker/scanning_grid.h>

namespace TLD {

    using BinaryDescriptor = uint16_t;

    class FernFeatureExtractor {
    public:
        FernFeatureExtractor(const ScanningGrid& grid);
        BinaryDescriptor GetDescriptor(cv::Mat& frame, cv::Size position, size_t scale_id) const;
        BinaryDescriptor GetDescriptor(cv::Mat& frame, cv::Rect bbox) const;
        BinaryDescriptor GetDescriptor(cv::Mat& frame) const;
        BinaryDescriptor operator()(cv::Mat& frame, cv::Size position, size_t scale_id) const;
    private:
        const ScanningGrid& _grid;
    };

}
