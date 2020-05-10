#include <tracker/utils.h>

std::ostream& operator<<(std::ostream &os, const cv::Rect& rect) {
    os << "Rect: " << rect.x << " " << rect.y << " "
       << rect.width << " " << rect.height << std::endl;
    return os;
}

double TLD::get_normalized_random() {
    return static_cast<double>(rand()) / RAND_MAX;
}

cv::Rect TLD::get_extended_rect_for_rotation(cv::Rect base_rect, double angle_degrees) {
    auto center_x = base_rect.x + 0.5*base_rect.width;
    auto center_y = base_rect.y + 0.5*base_rect.height;
    double angle_rad = abs(angle_degrees * M_PI / 180.0);
    auto size_x = static_cast<int>(base_rect.width * (cos(angle_rad) + sin(angle_rad)));
    auto size_y = static_cast<int>(base_rect.height * (cos(angle_rad) + sin(angle_rad)));
    auto x = static_cast<int>(center_x - 0.5*size_x);
    auto y = static_cast<int>(center_y - 0.5*size_y);
    return {x, y, size_x, size_y};
}

cv::Mat TLD::get_rotated_subframe(cv::Mat frame, cv::Rect subframe_rect, double angle) {
    cv::Mat rotated;
    auto extended_rect = get_extended_rect_for_rotation(subframe_rect, angle);
    cv::Mat src_subframe = frame(extended_rect);
    cv::Point2d pt(src_subframe.cols/2., src_subframe.rows/2.);
    cv::Mat r = getRotationMatrix2D(pt, angle, 1.0);
    cv::warpAffine(src_subframe, rotated, r, cv::Size(src_subframe.cols, src_subframe.rows));
    int offset_x = static_cast<int>(0.5*(extended_rect.width - subframe_rect.width));
    int offset_y = static_cast<int>(0.5*(extended_rect.height- subframe_rect.height));
    return rotated({offset_x, offset_y, extended_rect.width, extended_rect.height});
}

void TLD::rotate_subframe(cv::Mat& frame, cv::Rect subframe_rect, double angle) {
    cv::Mat rotated;
    auto extended_rect = get_extended_rect_for_rotation(subframe_rect, angle);
    cv::Mat src_subframe = frame(extended_rect);
    cv::Point2d pt(src_subframe.cols/2., src_subframe.rows/2.);
    cv::Mat r = getRotationMatrix2D(pt, angle, 1.0);
    cv::warpAffine(src_subframe, rotated, r, cv::Size(src_subframe.cols, src_subframe.rows));
    int offset_x = static_cast<int>(0.5*(extended_rect.width - subframe_rect.width));
    int offset_y = static_cast<int>(0.5*(extended_rect.height- subframe_rect.height));

    for (auto j = 0; j < subframe_rect.height; j++) {
        for (auto i = 0; i < subframe_rect.width; i++) {
            frame.at<uchar>(j + subframe_rect.y, i + subframe_rect.x) = rotated.at<uchar>(offset_y+j, offset_x+i);
        }
    }
}
