#pragma once

#include <vector>

#include <tracker/common.h>

namespace TLD {

    class Fern {
    public:
        Fern(size_t pairs_cnt);
        template<typename It>
        Fern(It begin, It end);

        std::vector<AbsFernPair> Transform(cv::Size base_bbox_size) const;

    private:
        std::vector<NormFernPair> _pairs;
    };

    template<typename It>
    Fern::Fern(It begin, It end) {
        _pairs = std::vector<NormFernPair>(begin, end);
    }

}
