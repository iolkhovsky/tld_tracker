#pragma once

#include <opencv2/opencv.hpp>

#include <tracker/fern.h>

namespace TLD {

    using PixelIdPair = std::pair<size_t, size_t>;

    class ScanningGrid {
    public:
        ScanningGrid(cv::Size frame_size) :
            _frame_size(frame_size),
            _fern(10) {
        }
        void SetBase(cv::Size bbox, double overlap, std::vector<double> scales);
        std::vector<cv::Size> GetPositionsCnt();
        std::vector<PixelIdPair> GetPixelPairs(cv::Size position, size_t scale_idx);

    private:
        cv::Size _frame_size;
        Fern _fern;
        cv::Size _base_bbox;
        std::vector<double> _scales;
        double _overlap;

        std::vector<std::vector<PixelIdPair>> _zero_shifted;
    };

}
