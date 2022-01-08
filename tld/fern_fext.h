#pragma once

#include "common.h"

#include "scanning_grid.h"

namespace tld {

    class FernFeatureExtractor {
    public:
        FernFeatureExtractor(std::shared_ptr<ScanningGrid> grid);
        BinaryDescriptor GetDescriptor(cv::Mat& frame, cv::Size position, size_t scale_id) const;
        BinaryDescriptor GetDescriptor(cv::Mat& frame, cv::Rect bbox) const;
        BinaryDescriptor GetDescriptor(cv::Mat& frame) const;
        BinaryDescriptor operator()(cv::Mat& frame, cv::Size position, size_t scale_id) const;
    private:
         std::shared_ptr<ScanningGrid> _grid;
    };

}
