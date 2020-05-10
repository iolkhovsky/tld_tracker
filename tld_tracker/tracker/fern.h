#pragma once

#include <cstdlib>
#include <vector>
#include <opencv2/opencv.hpp>
#include <utility>

namespace TLD {

    using NormFernPair = std::tuple<cv::Point2d, cv::Point2d>;
    using AbsFernPair = std::tuple<cv::Point2i, cv::Point2i>;

    class Fern {
    public:
        Fern(size_t pairs_cnt);
        template<typename It>
        Fern(It begin, It end);

        std::vector<AbsFernPair> Transform(cv::Size base_bbox_size);

    private:
        std::vector<NormFernPair> _pairs;
    };

    template<typename It>
    Fern::Fern(It begin, It end) {
        _pairs = std::vector<NormFernPair>(begin, end);
    }

}
