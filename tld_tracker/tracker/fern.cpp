#include <tracker/fern.h>
#include <tracker/tld_utils.h>

namespace TLD {

    Fern::Fern(size_t pairs_cnt) {
        _pairs.resize(pairs_cnt);
        for (auto &[p1, p2]: _pairs) {
            p1.x = get_normalized_random();
            p1.y = get_normalized_random();
            auto orientation = get_normalized_random();
            auto aux = get_normalized_random();
            if (orientation > 0.5) {
                p2.x = p1.x;
                p2.y = aux;
            } else {
                p2.y = p1.y;
                p2.x = aux;
            }
        }
    }

    Fern::Fern(const Fern& other)
        : _pairs(other._pairs) {
    }

    std::vector<AbsFernPair> Fern::Transform(cv::Size base_bbox_size) const {
        std::vector<AbsFernPair> out;
        auto img_w = base_bbox_size.width;
        auto img_h = base_bbox_size.height;
        for (const auto &[p1, p2]: _pairs) {
            std::pair<cv::Point2i, cv::Point2i> abs_pair;
            abs_pair.first = {static_cast<int>(p1.x * img_w), static_cast<int>(p1.y * img_h)};
            abs_pair.second = {static_cast<int>(p2.x * img_w), static_cast<int>(p2.y * img_h)};
            out.emplace_back(std::move(abs_pair));
        }
        return out;
    }

    size_t Fern::GetPairsCnt() const {
        return _pairs.size();
    }

}


