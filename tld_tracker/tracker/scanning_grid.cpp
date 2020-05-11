#include <tracker/scanning_grid.h>

namespace  {
    constexpr const char* AREA_ERROR = "ScanningGrid has received zero area base bbox!";
    constexpr const char* OVERLAP_ERROR = "ScanningGrid has received zero or negative overlap!";
    constexpr const char* SCALE_ERROR = "ScanningGrid has received zero or negative base scale!";
    constexpr const char* LARGE_SCALE_ERROR = "ScanningGrid scale larger than image!";
}

namespace TLD {

    void ScanningGrid::SetBase(cv::Size bbox, double overlap, std::vector<double> scales) {
        _base_bbox = bbox;
        _scales = scales;
        _overlap = overlap;

        _zero_shifted.clear();

        if (_base_bbox.area() <= 0)
            throw std::runtime_error(AREA_ERROR);
        if (overlap <= 0.0)
            throw std::runtime_error(OVERLAP_ERROR);

        for (auto scale: scales) {
            if (scale <= 0.0)
                throw std::runtime_error(SCALE_ERROR);

            cv::Size scaled_bbox(static_cast<int>(_base_bbox.width * scale),
                                 static_cast<int>(_base_bbox.height * scale));
            if ((scaled_bbox.width <= _frame_size.width) &&
                    (scaled_bbox.height <= _frame_size.height)) {
                std::vector<PixelIdPair> scale_points;

                auto fern_base = _fern.Transform(scaled_bbox);

                for (auto& [p1, p2]: fern_base) {
                    size_t offset0 = static_cast<size_t>(p1.x + p1.y * _frame_size.width);
                    size_t offset1 = static_cast<size_t>(p2.x + p2.y * _frame_size.width);
                    scale_points.push_back({offset0, offset1});
                }

                _zero_shifted.push_back(scale_points);
            } else
                throw std::runtime_error(LARGE_SCALE_ERROR);
        }
    }

    std::vector<cv::Size> ScanningGrid::GetPositionsCnt() const {
        std::vector<cv::Size> out;

        for (auto scale: _scales) {
            cv::Size grid_size;
            cv::Size scaled_bbox(static_cast<int>(_base_bbox.width * scale),
                                 static_cast<int>(_base_bbox.height * scale));

            int scanning_area_x = _frame_size.width - scaled_bbox.width;
            int scanning_area_y = _frame_size.height - scaled_bbox.height;
            int step_x = static_cast<int>(scaled_bbox.width * _overlap);
            int step_y = static_cast<int>(scaled_bbox.width * _overlap);

            grid_size.width = 1 + scanning_area_x / step_x;
            grid_size.height = 1 + scanning_area_y / step_y;

            out.push_back(grid_size);
        }

        return out;
    }

    std::vector<PixelIdPair> ScanningGrid::GetPixelPairs(cv::Size position, size_t scale_idx) const {
        std::vector<PixelIdPair> base = _zero_shifted[scale_idx];
        auto scale = _scales[scale_idx];
        cv::Size scaled_bbox(static_cast<int>(_base_bbox.width * scale),
                             static_cast<int>(_base_bbox.height * scale));
        int step_x = static_cast<int>(scaled_bbox.width * _overlap);
        int step_y = static_cast<int>(scaled_bbox.width * _overlap);

        int x_offset = position.width * step_x;
        int y_offset = position.height * step_y;
        int linear_offset = x_offset + y_offset * _frame_size.width;

        for (auto& [p1_idx, p2_idx]: base) {
            p1_idx += static_cast<size_t>(linear_offset);
            p2_idx += static_cast<size_t>(linear_offset);
        }

        return base;
    }

    std::vector<PixelIdPair> ScanningGrid::GetPixelPairs(cv::Rect bbox) const {
        std::vector<PixelIdPair> out;

        cv::Size rect_size(bbox.width, bbox.height);
        std::vector<AbsFernPair> local_coords = _fern.Transform(rect_size);

        for (auto& [p1, p2]: local_coords) {
            auto p1_offset = (p1.x + bbox.x) + (p1.y + bbox.y) * _frame_size.width;
            auto p2_offset = (p2.x + bbox.x) + (p2.y + bbox.y) * _frame_size.width;
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

}
