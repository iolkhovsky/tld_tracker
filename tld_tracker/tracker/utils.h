#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdlib>

std::ostream& operator<<(std::ostream &os, const cv::Rect& rect);

namespace TLD {

    enum class ProposalSource {
        tracker,
        detector
    };

    struct Candidate {
        cv::Rect strobe;
        double prob;
        bool valid;
        bool training;
        ProposalSource src;
    };

    double get_normalized_random();

}
