#include <tracker/opt_flow_tracker.h>

namespace TLD {

    OptFlowTracker::OptFlowTracker() {

    }

    void OptFlowTracker::SetFrame(cv::Mat frame) {
        _prev_frame = std::move(_current_frame);
        _current_frame = std::move(frame);
    }

    void OptFlowTracker::SetTarget(cv::Rect strobe) {
        _target = strobe;
    }

    Candidate OptFlowTracker::Track() {
        Candidate out;
        out.src = ProposalSource::tracker;
        if ((_target.area() == 0) || _prev_frame.empty() || _current_frame.empty())
            return out;
        cv::Mat mask = cv::Mat(_prev_frame.rows, _prev_frame.cols, CV_8UC1);
        mask.setTo(0);
        cv::Mat target_subframe = mask(_target);
        target_subframe.setTo(255);

        // find good features on the previous frame
        cv::goodFeaturesToTrack(_prev_frame, _prev_points, 100, 0.01, 10.0, mask, 7, false, 0.04);
        // refine them
        if(!_prev_points.empty()) {
           cornerSubPix(_prev_frame, _prev_points, cv::Size(3,3), cv::Size(-1,-1),
                        cv::TermCriteria(cv::TermCriteria::COUNT, 100, 0.01));
        }
        // if there are some feature points on the previous frame
        if(!_prev_points.empty()) {
            // start to compute optflow in 2 directions
            std::vector<float> err;
            cv::calcOpticalFlowPyrLK(_prev_frame, _current_frame, _prev_points, _cur_points,
                                 _cur_status, err, cv::Size(5,5), 3,
                                 cv::TermCriteria(cv::TermCriteria::COUNT, 100, 0.01), 0, 0.001);
            cv::calcOpticalFlowPyrLK(_current_frame, _prev_frame, _cur_points, _backtrace,
                                 _backtrace_status, err, cv::Size(5,5), 3,
                                 cv::TermCriteria(cv::TermCriteria::COUNT, 100, 0.01), 0, 0.001);
            // assemble output point pairs
            _prev_out_points.clear();
            _cur_out_points.clear();
            for(size_t i = 0; i < _prev_points.size(); i++) {
                if (_cur_status[i] && _backtrace_status[i]) {
                    if (cv::norm(_prev_points[i] - _backtrace[i]) <= 1) {
                        _prev_out_points.push_back(_prev_points[i]);
                        _cur_out_points.push_back(_cur_points[i]);
                    } else
                        continue;
                } else
                    continue;
            }
        }
        cv::Point2d mean_shift, mean_scale;
        if (_prev_out_points.size() > 3) {// can trust to results
            mean_shift = get_mean_shift(_prev_out_points, _cur_out_points);
            mean_scale = get_scale(_prev_out_points, _cur_out_points);

            out.prob = static_cast<double>(_prev_out_points.size()) / _prev_points.size();
            out.valid = (_prev_out_points.size() >= 3) && (_prev_points.size() > 10);
            out.strobe.x = _target.x + static_cast<int>(mean_shift.x);
            out.strobe.y = _target.y + static_cast<int>(mean_shift.y);
            out.strobe.width = _target.width;
            out.strobe.height = _target.height;

            int abs_w_half_change = (static_cast<int>(mean_scale.x * _target.width) - _target.width) / 2;
            int abs_h_half_change = (static_cast<int>(mean_scale.y * _target.height) - _target.height) / 2;
            out.strobe.x -= abs_w_half_change;
            out.strobe.width += 2*abs_w_half_change;
            out.strobe.y -= abs_h_half_change;
            out.strobe.height += 2*abs_h_half_change;
        } else {
            out.valid = false;
        }
        return out;
    }

}


