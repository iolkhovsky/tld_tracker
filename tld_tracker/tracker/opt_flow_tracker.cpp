#include <tracker/opt_flow_tracker.h>

namespace TLD {

OptFlowTracker::OptFlowTracker() {

}

void OptFlowTracker::SetFrame(cv::Mat& frame) {
    _prev_frame = std::move(_current_frame);
}

void OptFlowTracker::SetTarget(cv::Rect strobe) {

}

Candidate OptFlowTracker::Track() {

}

}


