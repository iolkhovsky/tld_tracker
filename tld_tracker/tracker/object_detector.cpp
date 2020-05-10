#include <tracker/object_detector.h>

namespace TLD {

ObjectDetector::ObjectDetector() {

}

void ObjectDetector::SetFrame(std::shared_ptr<cv::Mat> img) {
    _frame_ptr = img;
}

void ObjectDetector::SetTarget(cv::Rect strobe) {

}

std::vector<Candidate> ObjectDetector::Detect() {

}

void ObjectDetector::Train(Candidate prediction) {

}

}
