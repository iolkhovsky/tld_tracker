#include <tracker/fern.h>
#include <tracker/utils.h>

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

    std::vector<AbsFernPair> Fern::Transform(cv::Size base_bbox_size) const {
        std::vector<AbsFernPair> out;
        auto img_w = base_bbox_size.width;
        auto img_h = base_bbox_size.height;
        for (auto &[p1, p2]: _pairs) {
            auto abs_pair = std::make_pair<cv::Point2i, cv::Point2i>(
                {static_cast<int>(p1.x * img_w), static_cast<int>(p1.y * img_h)},
                {static_cast<int>(p2.x * img_w), static_cast<int>(p2.y * img_h)});
            out.emplace_back(std::move(abs_pair));
        }
        return out;
    }

}


