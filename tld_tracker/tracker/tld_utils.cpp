#include <tracker/tld_utils.h>
#include <sstream>

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

uint8_t TLD::bilinear_interp_for_point(double x, double y, const cv::Mat& frame)
{
    uint8_t* data = frame.data;
    int image_x_size = frame.cols;
    int image_y_size = frame.rows;

    x = std::min(static_cast<double>(image_x_size), std::max(0.0, x));
    y = std::min(static_cast<double>(image_y_size), std::max(0.0, y));

    int x1 = static_cast<int>(x);
    int x2 = static_cast<int>(x1+1);
    int y1 = static_cast<int>(y);
    int y2 = static_cast<int>(y1+1);

    int Ix1y1 = data[y1*image_x_size + x1];
    int Ix2y1 = data[y1*image_x_size + x2];
    int Ix1y2 = data[y2*image_x_size + x1];
    int Ix2y2 = data[y2*image_x_size + x2];

    double p1 = Ix1y1 + (x-x1)*(Ix2y1-Ix2y1)/(x2-x1);
    double p2 = Ix1y2 + (x-x1)*(Ix2y2-Ix2y2)/(x2-x1);
    int res = static_cast<int>(p1 + (y-y1)*(p2-p1));

    uint8_t out = static_cast<uint8_t>(std::min(255, std::max(0, res)));
    return out;
}

double TLD::degree2rad(double angle) {
    return angle * M_PI / 180.0;
}

double TLD::rad2degree(double angle) {
    return angle * 180.0/ M_PI ;
}

cv::Mat TLD::subframe_linear_transform(const cv::Mat& frame, cv::Rect strobe, double angle,
                                    double scale, int offset_x, int offset_y) {
    cv::Mat out = frame(strobe).clone();
    uint8_t* out_data = out.data;
    angle = degree2rad(angle);
    int central_x_pix = strobe.x + strobe.width / 2;
    int central_y_pix = strobe.y + strobe.height / 2;
    int offs_x = strobe.x - central_x_pix;
    int offs_y = strobe.y - central_y_pix;

    double x_scaled,y_scaled,snan,csan,x_rotated,y_rotated;

    for (int j=0;j<strobe.height;j++) {
        for (int i=0;i<strobe.width;i++) {
            if (scale!=1.0) {
                x_scaled = (offs_x+i)*(1/scale) + central_x_pix;
                y_scaled = (offs_y+j)*(1/scale) + central_y_pix;
            } else {
                x_scaled = strobe.x+i;
                y_scaled = strobe.y+j;
            }

            if (abs(angle) > 1e-9) {
                snan = sin(angle);
                csan = cos(angle);
                x_rotated = (x_scaled-central_x_pix)*csan + (y_scaled-central_y_pix)*snan + central_x_pix;
                y_rotated = (-1)*(x_scaled-central_x_pix)*snan + (y_scaled-central_y_pix)*csan + central_y_pix;
            } else {
                x_rotated = x_scaled;
                y_rotated = y_scaled;
            }

            x_rotated = x_rotated - offset_x;
            y_rotated = y_rotated - offset_y;

            out_data[j*strobe.width+i]=bilinear_interp_for_point(x_rotated,y_rotated,frame);
         }
    }

    return out;
}

double TLD::iou(cv::Rect a, cv::Rect b) {
    int intersection_x_min = std::max(a.x, b.x);
    int intersection_y_min = std::max(a.y, b.y);
    int intersection_x_max = std::min(a.x + a.width, b.x + b.width);
    int intersection_y_max = std::max(a.y + a.height, b.y + b.height);

    if ((intersection_x_max <= intersection_x_min) ||
            (intersection_y_max <= intersection_y_min))
        return 0.0;
    else {
        double intersection_area = (intersection_y_max - intersection_y_min) *
                (intersection_x_max - intersection_x_min);
        return intersection_area / (a.area() + b.area() - intersection_area);
    }
}


cv::Point2f TLD::get_mean_shift(const std::vector<cv::Point2f> &start, const std::vector<cv::Point2f> &stop)
{
    cv::Point2f acc(0.0f, 0.0f);
    for(size_t i = 0; i < start.size(); i++)
        acc += stop[i] - start[i];
    acc = acc / double(start.size());
    return acc;
}

void TLD::drawCandidate(cv::Mat& frame, Candidate candidate) {
    cv::Point2i p1(candidate.strobe.x, candidate.strobe.y);
    cv::Point2i p2(candidate.strobe.x + candidate.strobe.width - 1,
                   candidate.strobe.y + candidate.strobe.height - 1);
    cv::Scalar color = CV_RGB(0,0,0);
    if (candidate.src == ProposalSource::tracker)
        color = CV_RGB(0,255,0);
    if (candidate.src == ProposalSource::detector)
        color = CV_RGB(0,0,255);
    cv::rectangle(frame, p1, p2, color);
    std::stringstream ss;
    ss << "Prob: ";
    ss.precision(2);
    ss << candidate.prob;
    cv::Point2i org(candidate.strobe.x, candidate.strobe.y - 10);
    cv::putText(frame, ss.str().c_str(), org, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
}


