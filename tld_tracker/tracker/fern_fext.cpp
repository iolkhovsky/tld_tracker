#include <tracker/fern_fext.h>

namespace TLD {

    FernFeatureExtractor::FernFeatureExtractor(std::shared_ptr<ScanningGrid> grid)
        : _grid(grid) {
    }

    BinaryDescriptor FernFeatureExtractor::GetDescriptor(cv::Mat& frame, cv::Size position, size_t scale_id) const {
        BinaryDescriptor desc = 0x0;
        unsigned char *frame_data = frame.data;
        auto pairs = _grid->GetPixelPairs(frame, position, scale_id);

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
        auto pairs = _grid->GetPixelPairs(frame, bbox);

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
        auto pairs = _grid->GetPixelPairs(frame);

        size_t mask = 0x1;
        for (auto& [p1, p2]: pairs) {
            if (frame_data[p1] > frame_data[p2])
                desc |= mask;
            mask = mask << 1;
        }

        return desc;
    }

    BinaryDescriptor FernFeatureExtractor::operator()(cv::Mat& frame, cv::Size position, size_t scale_id) const {
        return GetDescriptor(frame, position, scale_id);
    }

}
