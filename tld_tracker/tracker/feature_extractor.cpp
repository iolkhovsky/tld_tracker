#include <tracker/feature_extractor.h>

namespace TLD {

    BinaryDescriptor FeatureExtractor::GetDescriptor(cv::Mat& frame, std::vector<PixelIdPair> pairs) {
        BinaryDescriptor desc = 0x0;
        unsigned char *frame_data = frame.data;

        size_t mask = 0x1;
        for (auto& [p1, p2]: pairs) {
            if (frame_data[p1] > frame_data[p2])
                desc |= mask;
            mask = mask << 1;
        }

        return desc;
    }

    BinaryDescriptor FeatureExtractor::operator()(cv::Mat& frame, std::vector<PixelIdPair> pairs) {
        return GetDescriptor(frame, pairs);
    }

}
