#pragma once

#include <tracker/common.h>
#include <tracker/fern.h>
#include <tracker/tld_utils.h>

namespace TLD {

    class ScanningGrid {
    public:
        ScanningGrid(cv::Size frame_size);
        ScanningGrid(const ScanningGrid& other);
        ScanningGrid(ScanningGrid&& other);
        void SetBase(cv::Size bbox, double overlap, std::vector<double> scales);
        std::vector<cv::Size> GetPositionsCnt() const;
        std::vector<PixelIdPair> GetPixelPairs(cv::Size position, size_t scale_idx) const;
        std::vector<PixelIdPair> GetPixelPairs(cv::Rect bbox) const;
        const Fern& GetFern() const;
        cv::Size GetOverlap() const;
        cv::Size FetFrameSize() const;

    private:
        cv::Size _frame_size;
        Fern _fern;
        cv::Size _base_bbox;
        std::vector<double> _scales;
        double _overlap;

        std::vector<std::vector<PixelIdPair>> _zero_shifted;
    };

}
