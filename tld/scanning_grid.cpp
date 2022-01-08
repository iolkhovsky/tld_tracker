#include "scanning_grid.h"

namespace  {
    constexpr const char* AREA_ERROR = "ScanningGrid has received zero area base bbox!";
    constexpr const char* OVERLAP_ERROR = "ScanningGrid has received zero or negative overlap!";
    constexpr const char* SCALE_ERROR = "ScanningGrid has received zero or negative base scale!";
    constexpr const char* LARGE_SCALE_ERROR = "ScanningGrid scale larger than image!";
}

namespace tld {

    ScanningGrid::ScanningGrid(cv::Size frame_size) :
        _frame_size(frame_size),
        _fern(BINARY_DESCRIPTOR_WIDTH) {
    }

    ScanningGrid::ScanningGrid(const ScanningGrid& other)
        : _frame_size(other.FetFrameSize()),
          _fern(other.GetFern()) {
        _base_bbox=other._base_bbox;
        _scales = {other._scales.begin(), other._scales.begin()};
        _overlap = other._overlap;
        _zero_shifted = {other._zero_shifted.begin(), other._zero_shifted.end()};
    }

    ScanningGrid::ScanningGrid(ScanningGrid&& other)
        : _fern(other.GetFern()) {
        _frame_size = other.FetFrameSize();
        _base_bbox = std::move(other._base_bbox);
        _scales = std::move(other._scales);
        _overlap = other._overlap;
        _zero_shifted = std::move(other._zero_shifted);
    }

    void ScanningGrid::SetBase(cv::Size bbox, double overlap, std::vector<double> scales) {
        _base_bbox = bbox;
        _scales = scales;
        _overlap = overlap;

        _zero_shifted.clear();
        _steps.clear();
        _bbox_sizes.clear();

        if (_base_bbox.area() <= 0)
            throw std::runtime_error(AREA_ERROR);
        if (overlap <= 0.0)
            throw std::runtime_error(OVERLAP_ERROR);

        for (auto scale: scales) {
            if (scale <= 0.0)
                throw std::runtime_error(SCALE_ERROR);

            cv::Size scaled_bbox(static_cast<int>(_base_bbox.width * scale),
                                 static_cast<int>(_base_bbox.height * scale));
            _bbox_sizes.push_back(scaled_bbox);
            if ((scaled_bbox.width <= _frame_size.width) &&
                    (scaled_bbox.height <= _frame_size.height)) {
                std::vector<PixelIdPair> scale_points;

                auto fern_base = _fern.Transform(scaled_bbox);

                for (const auto& [p1, p2]: fern_base) {
                    std::pair<size_t, size_t> _point_pair;
                    size_t offset0 = static_cast<size_t>(p1.x + p1.y * _frame_size.width);
                    size_t offset1 = static_cast<size_t>(p2.x + p2.y * _frame_size.width);
                    _point_pair.first = offset0;
                    _point_pair.second = offset1;
                    scale_points.push_back(std::move(_point_pair));
                }

                _zero_shifted.push_back(std::move(scale_points));
                int step_x = static_cast<int>(scaled_bbox.width * _overlap);
                int step_y = static_cast<int>(scaled_bbox.height * _overlap);
                step_x = std::max(4, step_x);
                step_y = std::max(4, step_y);
                _steps.push_back({step_x, step_y});
            } else
                throw std::runtime_error(LARGE_SCALE_ERROR);
        }
    }

    std::vector<cv::Size> ScanningGrid::GetPositionsCnt() const {
        return get_scan_position_cnt(_frame_size, _base_bbox, _scales, _steps);
    }

    std::vector<PixelIdPair> ScanningGrid::GetPixelPairs(const cv::Mat& frame, cv::Size position, size_t scale_idx) const {
        if (_zero_shifted.size() == 0)
            throw std::runtime_error("Reference grid is ampty!");
        std::vector<PixelIdPair> base = _zero_shifted[scale_idx];
        auto scale = _scales[scale_idx];
        cv::Size scaled_bbox(static_cast<int>(_base_bbox.width * scale),
                             static_cast<int>(_base_bbox.height * scale));
        int step_x = _steps.at(scale_idx).width;
        int step_y = _steps.at(scale_idx).height;

        int x_offset = position.width * step_x;
        int y_offset = position.height * step_y;
        int linear_offset = x_offset + y_offset * frame.step[0];

        for (auto& [p1_idx, p2_idx]: base) {
            p1_idx += static_cast<size_t>(linear_offset);
            p2_idx += static_cast<size_t>(linear_offset);
        }

        return base;
    }

    std::vector<PixelIdPair> ScanningGrid::GetPixelPairs(const cv::Mat& frame, cv::Rect bbox) const {
        std::vector<PixelIdPair> out;

        cv::Size rect_size(bbox.width, bbox.height);
        std::vector<AbsFernPair> local_coords = _fern.Transform(rect_size);

        for (auto& [p1, p2]: local_coords) {
            auto p1_offset = (p1.x + bbox.x) + (p1.y + bbox.y) * frame.step[0];
            auto p2_offset = (p2.x + bbox.x) + (p2.y + bbox.y) * frame.step[0];
            out.push_back({p1_offset, p2_offset});
        }

        return out;
    }

    std::vector<PixelIdPair> ScanningGrid::GetPixelPairs(const cv::Mat& frame) const {
        std::vector<PixelIdPair> out;

        cv::Size rect_size(frame.cols, frame.rows);
        std::vector<AbsFernPair> local_coords = _fern.Transform(rect_size);

        for (auto& [p1, p2]: local_coords) {
            auto p1_offset = p1.x + p1.y * frame.step[0];
            auto p2_offset = p2.x + p2.y * frame.step[0];
            out.push_back({p1_offset, p2_offset});
        }

        return out;
    }

    const Fern& ScanningGrid::GetFern() const {
        return _fern;
    }

    cv::Size ScanningGrid::GetOverlap() const {
        cv::Size out;
        out.width = static_cast<int>(_overlap * _base_bbox.width);
        out.height = static_cast<int>(_overlap * _base_bbox.height);
        return out;
    }

    cv::Size ScanningGrid::FetFrameSize() const {
        return _frame_size;
    }

    std::vector<double> ScanningGrid::GetScales() const {
        return _scales;
    }

    std::vector<cv::Size> ScanningGrid::GetSteps() const {
        return _steps;
    }

    std::vector<cv::Size> ScanningGrid::GetBBoxSizes() const {
        return _bbox_sizes;
    }

}
