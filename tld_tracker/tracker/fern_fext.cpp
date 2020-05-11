#include <tracker/fern_fext.h>

namespace TLD {

    FernFeatureExtractor::FernFeatureExtractor(const ScanningGrid& grid) :
        _grid(grid) {
    }

    BinaryDescriptor FernFeatureExtractor::GetDescriptor(cv::Mat& frame, cv::Size position, size_t scale_id) const {
        BinaryDescriptor desc = 0x0;
        unsigned char *frame_data = frame.data;
        auto pairs = _grid.GetPixelPairs(position, scale_id);

        size_t mask = 0x1;
        for (auto& [p1, p2]: pairs) {
            if (frame_data[p1] > frame_data[p2])
                desc |= mask;
            mask = mask << 1;
        }

        return desc;
    }

    BinaryDescriptor FernFeatureExtractor::GetDescriptor(cv::Mat& frame, cv::Rect bbox) const {
        BinaryDescriptor desc = 0x0;
        unsigned char *frame_data = frame.data;
        auto pairs = _grid.GetPixelPairs(bbox);

        size_t mask = 0x1;
        for (auto& [p1, p2]: pairs) {
            if (frame_data[p1] > frame_data[p2])
                desc |= mask;
            mask = mask << 1;
        }

        return desc;
    }

    BinaryDescriptor FernFeatureExtractor::GetDescriptor(cv::Mat& frame) const {
        BinaryDescriptor desc = 0x0;

        unsigned char *frame_data = frame.data;
        const auto& fern = _grid.GetFern();

        std::vector<AbsFernPair> abs_coord_pairs = fern.Transform({frame.cols, frame.rows});

        size_t mask = 0x1;
        for (auto& [p1, p2]: abs_coord_pairs) {
            auto p1offset = p1.x + p1.y * frame.cols;
            auto p2offset = p2.x + p2.y * frame.cols;
            if (frame_data[p1offset] > frame_data[p2offset])
                desc |= mask;
            mask = mask << 1;
        }

        return desc;
    }

    BinaryDescriptor FernFeatureExtractor::operator()(cv::Mat& frame, cv::Size position, size_t scale_id) const {
        return GetDescriptor(frame, position, scale_id);
    }

}
