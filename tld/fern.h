#pragma once

#include <vector>

#include "common.h"

namespace tld {

    class Fern {
    public:
        Fern(size_t pairs_cnt);
        template<typename It>
        Fern(It begin, It end);
        Fern(const Fern& other);
        size_t GetPairsCnt() const;
        std::vector<AbsFernPair> Transform(cv::Size base_bbox_size) const;

    private:
        std::vector<NormFernPair> _pairs;
    };

    template<typename It>
    Fern::Fern(It begin, It end) {
        _pairs = std::vector<NormFernPair>(begin, end);
    }

}
