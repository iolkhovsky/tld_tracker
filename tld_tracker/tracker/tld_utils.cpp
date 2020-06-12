#include <tracker/tld_utils.h>
#include <sstream>

std::ostream& operator<<(std::ostream &os, const cv::Rect& rect) {
    os << "Rect: " << rect.x << " " << rect.y << " "
       << rect.width << " " << rect.height << std::endl;
    return os;
}

bool tld::Candidate::operator<(const Candidate& other) const {
    return prob < other.prob;
}

double tld::get_normalized_random() {
    return static_cast<double>(rand()) / RAND_MAX;
}

int tld::get_random_int(int maxint) {
    return rand() % (maxint + 1);
}

cv::Mat tld::generate_random_image() {
    cv::Mat out(cv::Size(640, 480), CV_8UC1);
    for (int j = 0; j < out.rows; j++)
        for (int i = 0; i < out.cols; i++)
            out.at<uint8_t>(j, i) = get_random_int(255);
    return out;
}

cv::Mat tld::generate_random_image(cv::Size sz) {
    cv::Mat out(cv::Size(sz.width, sz.height), CV_8UC1);
    for (int j = 0; j < out.rows; j++)
        for (int i = 0; i < out.cols; i++)
            out.at<uint8_t>(j, i) = get_random_int(255);
    return out;
}

cv::Rect tld::get_extended_rect_for_rotation(cv::Rect base_rect, double angle_degrees) {
    auto center_x = base_rect.x + 0.5*base_rect.width;
    auto center_y = base_rect.y + 0.5*base_rect.height;
    double angle_rad = abs(angle_degrees * M_PI / 180.0);
    auto size_x = static_cast<int>(base_rect.width * (cos(angle_rad) + sin(angle_rad)));
    auto size_y = static_cast<int>(base_rect.height * (cos(angle_rad) + sin(angle_rad)));
    auto x = static_cast<int>(center_x - 0.5*size_x);
    auto y = static_cast<int>(center_y - 0.5*size_y);
    return {x, y, size_x, size_y};
}

cv::Mat tld::get_rotated_subframe(cv::Mat frame, cv::Rect subframe_rect, double angle) {
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

void tld::rotate_subframe(cv::Mat& frame, cv::Rect subframe_rect, double angle) {
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

uint8_t tld::bilinear_interp_for_point(double x, double y, const cv::Mat& frame)
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

double tld::degree2rad(double angle) {
    return angle * M_PI / 180.0;
}

double tld::rad2degree(double angle) {
    return angle * 180.0/ M_PI ;
}

cv::Mat tld::subframe_linear_transform(const cv::Mat& frame, cv::Rect strobe, double angle,
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

double tld::compute_iou(cv::Rect a, cv::Rect b) {
    int intersection_x_min = std::max(a.x, b.x);
    int intersection_y_min = std::max(a.y, b.y);
    int intersection_x_max = std::min(a.x + a.width, b.x + b.width);
    int intersection_y_max = std::min(a.y + a.height, b.y + b.height);

    if ((intersection_x_max <= intersection_x_min) ||
            (intersection_y_max <= intersection_y_min))
        return 0.0;
    else {
        double intersection_area = (intersection_y_max - intersection_y_min) *
                (intersection_x_max - intersection_x_min);
        return intersection_area / (a.area() + b.area() - intersection_area);
    }
}


cv::Point2f tld::get_mean_shift(const std::vector<cv::Point2f> &start, const std::vector<cv::Point2f> &stop)
{
    cv::Point2f acc(0.0f, 0.0f);
    for(size_t i = 0; i < start.size(); i++)
        acc += stop[i] - start[i];
    acc = acc / double(start.size());
    return acc;
}

double tld::get_scale(const std::vector<cv::Point2f> &start, const std::vector<cv::Point2f> &stop) {
    std::vector<double> scale_sample;

    for (size_t i = 0; i < start.size(); i++) {
        for (size_t j = i + 1; j < start.size(); j++) {
            double sq_dist_prev = std::pow(cv::norm(start[i] - start[j]),2);
            double sq_dist_cur = std::pow(cv::norm(stop[i] - stop[j]),2);
            scale_sample.push_back(std::sqrt(sq_dist_cur/sq_dist_prev));
        }
    }

    std::sort(scale_sample.begin(), scale_sample.end());
    return scale_sample[scale_sample.size() / 2];
}

void tld::drawCandidate(cv::Mat& frame, Candidate candidate) {
    cv::Point2i p1(candidate.strobe.x, candidate.strobe.y);
    cv::Point2i p2(candidate.strobe.x + candidate.strobe.width - 1,
                   candidate.strobe.y + candidate.strobe.height - 1);
    cv::Scalar color = CV_RGB(0,0,0);
    int thickness = 1;
    if (candidate.src == ProposalSource::tracker)
        color = CV_RGB(0,255,0);
    if (candidate.src == ProposalSource::detector)
        color = CV_RGB(0,0,255);
    if (candidate.src == ProposalSource::mixed)
        color = CV_RGB(0,255,255);
    if (candidate.src == ProposalSource::final) {
        if (candidate.valid == false)
            return;
        color = CV_RGB(255,0,0);
        thickness = static_cast<int>(4 * candidate.prob);
    }
    cv::rectangle(frame, p1, p2, color, thickness);
    std::stringstream ss;
    ss << "Prob: ";
    ss.precision(2);
    ss << candidate.prob;
    cv::Point2i org(candidate.strobe.x, candidate.strobe.y - 10);
    cv::putText(frame, ss.str().c_str(), org, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
}

void tld::drawCandidates(cv::Mat& frame, std::vector<Candidate> candidates) {
    for (auto& item: candidates)
        drawCandidate(frame, item);
}

std::vector<cv::Size> tld::get_scan_position_cnt(cv::Size frame_size, cv::Size box, std::vector<double> scales, std::vector<cv::Size> steps) {
    std::vector<cv::Size> out;

    size_t scale_id = 0;
    for (auto scale: scales) {
        cv::Size grid_size;
        cv::Size scaled_bbox(static_cast<int>(box.width * scale),
                             static_cast<int>(box.height * scale));

        int scanning_area_x = frame_size.width - scaled_bbox.width;
        int scanning_area_y = frame_size.height - scaled_bbox.height;
        int step_x = steps.at(scale_id).width;
        int step_y = steps.at(scale_id).height;

        grid_size.width = 1 + scanning_area_x / step_x;
        grid_size.height = 1 + scanning_area_y / step_y;

        out.push_back(grid_size);
        scale_id++;
     }

    return out;
}

double tld::images_correlation(const cv::Mat &image_1, const cv::Mat &image_2)   {
    cv::Mat im_float_1;
    image_1.convertTo(im_float_1, CV_32F);
    cv::Mat im_float_2;
    image_2.convertTo(im_float_2, CV_32F);

    int n_pixels = im_float_1.rows * im_float_1.cols;

    cv::Scalar im1_Mean, im1_Std, im2_Mean, im2_Std;
    cv::meanStdDev(im_float_1, im1_Mean, im1_Std);
    cv::meanStdDev(im_float_2, im2_Mean, im2_Std);

    double covar = (im_float_1 - im1_Mean).dot(im_float_2 - im2_Mean) / n_pixels;
    double correl = covar / (im1_Std[0] * im2_Std[0]);

    return correl;
}


namespace tld {
    class CandidateComparator {
    public:
        bool operator()(const Candidate& lhs, const Candidate& rhs) {
            return lhs.prob > rhs.prob;
        }
    };
}

std::vector<tld::Candidate> tld::non_max_suppression(const std::vector<Candidate>& in, double threshold_iou) {
    if (in.empty())
        return {};
    std::vector<Candidate> out;
    std::multiset<Candidate, CandidateComparator> sorted(begin(in), end(in));

    while (sorted.size()) {
        auto max = *sorted.begin();
        out.push_back(max);
        sorted.erase(sorted.begin());
        auto reference_rect = out.back().strobe;
        if (sorted.size()) {
            for (auto it = sorted.begin(); it != sorted.end(); ) {
                double iou = compute_iou(it->strobe, reference_rect);
                if (iou > threshold_iou) {
                    it = sorted.erase(it);
                } else {
                    it++;
                }
            }
        }
    }
    return out;
}

std::vector<tld::Candidate> tld::clusterize_candidates(const std::vector<Candidate>& in, double threshold_iou) {
    if (in.empty())
        return {};
    std::vector<Candidate> out;
    std::vector<std::vector<Candidate>> clusters;

    for (const auto &sample: in) {
        bool match = false;
        for (auto& cluster: clusters) {
            match = true;
            for (auto& item: cluster) {
                double iou = compute_iou(item.strobe, sample.strobe);
                if (iou < threshold_iou) {
                    match = false;
                    break;
                }
            }
            if (match)
                cluster.push_back(sample);
        }
        if (!match)
            clusters.push_back({sample});
    }

    for (auto& cluster: clusters) {
        Candidate avg = aggregate_candidates(cluster);
        avg.src = cluster.front().src;
        out.push_back(avg);
    }

    return out;
}

tld::Candidate tld::aggregate_candidates(std::vector<Candidate> sample) {
    Candidate out;
    out.prob = 0.0;
    out.aux_prob = 0.0;
    double x = 0.0;
    double y = 0.0;
    double w = 0.0;
    double h = 0.0;
    for (auto& example: sample) {
        x += example.strobe.x;
        y += example.strobe.y;
        w += example.strobe.width;
        h += example.strobe.height;
        out.prob = std::max(out.prob, example.prob);
        out.aux_prob = std::max(out.aux_prob, example.aux_prob);
    }
    out.strobe.x = static_cast<int>(x / sample.size());
    out.strobe.y = static_cast<int>(y / sample.size());
    out.strobe.width = static_cast<int>(w / sample.size());
    out.strobe.height = static_cast<int>(h / sample.size());
    out.src = ProposalSource::mixed;
    return out;
}

cv::Rect tld::adjust_rect_to_frame(cv::Rect rect, cv::Size sz) {
    cv::Rect out;
    int x1, y1, x2, y2;
    x1 = rect.x;
    y1 = rect.y;
    x2 = x1 + rect.width;
    y2 = y1 + rect.height;
    x1 = std::max(std::min(sz.width-1, x1), 0);
    y1 = std::max(std::min(sz.height-1, y1), 0);
    x2 = std::max(std::min(sz.width-1, x2), 0);
    y2 = std::max(std::min(sz.height-1, y2), 0);
    out.x = x1;
    out.y = y1;
    out.width = x2-x1;
    out.height = y2-y1;
    return out;
}

bool tld::strobe_is_outside(cv::Rect rect, cv::Size sz) {
    return (rect.x < 0) || (rect.x >= sz.width) ||
            (rect.y < 0) || (rect.y >= sz.height) ||
            (rect.x + rect.width < 0) || (rect.x + rect.width >= sz.width) ||
            (rect.y + rect.height < 0) || (rect.y + rect.height >= sz.height);
}

double tld::get_frame_std_dev(const cv::Mat& frame, cv::Rect roi) {
    cv::Mat variance, mean;
    cv::meanStdDev(frame(roi), mean, variance);
    double stddev = variance.at<double>(0,0);
    return stddev;
}



