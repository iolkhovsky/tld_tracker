#include <tracker/tld_tracker.h>

std::ostream& operator<<(std::ostream& os, const tld::TldTracker& tracker) {
    os << std::endl;
    os << "<TLD tracker object>" << std::endl;
    os << "Processing:\t" << (tracker.GetStatus().processing ? "enable" : "disable") << std::endl;
    os << "Target:\t" << (tracker.GetStatus().valid_object ? "valid" : "invalid") << std::endl;
    os << "Probability:\t" << tracker.GerCurrentPrediction().prob << std::endl;
    os << "Size (W/H):\t" << tracker.GerCurrentPrediction().strobe.width << "/"
       << tracker.GerCurrentPrediction().strobe.height<< std::endl;
    os << "Center (X/Y):\t" << tracker.GerCurrentPrediction().strobe.x << "/"
       << tracker.GerCurrentPrediction().strobe.y << std::endl;
    os << "Training status:\t" << (tracker.GetStatus().training ? "enable" : "disable") << std::endl;
    os << "Relocation flag:\t" << (tracker.GetStatus().tracker_relocation ? "enable" : "disable") << std::endl;
    os << "Detector proposals:\t" << tracker.GetStatus().detector_candidates_cnt << std::endl;
    os << "Detector clusters:\t" << tracker.GetStatus().detector_clusters_cnt << std::endl;
    os << "Status:\t\t" << tracker.GetStatus().message << std::endl;
    os << std::endl;
    return os;
}

namespace tld {

TldTracker make_tld_tracker() {
    Settings s;
    TldTracker tracker(s);
    return tracker;
}

TldTracker make_tld_tracker(Settings s) {
    TldTracker tracker(s);
    return tracker;
}

TldTracker::TldTracker(Settings settings)
    : _settings(settings), _integrator(_model) {
    _processing_en = false;
}

Candidate TldTracker::ProcessFrame(const cv::Mat& input_frame) {
    _src_frame = input_frame.clone();
    cv::blur(_src_frame, _lf_frame, cv::Size(14,14));

    _detector.SetFrame(std::make_shared<cv::Mat>(_lf_frame));
    _model.SetFrame(std::make_shared<cv::Mat>(_src_frame));
    _tracker.SetFrame(_src_frame);

    if (_processing_en) {
        _detector_proposals = _detector.Detect();
        _tracker_proposal = _tracker.Track();
        std::tie(_prediction, _training_en, _tracker_relocate) = _integrator.Integrate(_detector_proposals, _tracker_proposal);
        if (_training_en) {
            _detector.Train(_prediction);
            _detector.UpdateGrid(_prediction);
            _model.Train(_prediction);
        }
        if (_tracker_relocate)
            _tracker.SetTarget(_prediction.strobe);
        else
            _tracker.SetTarget(_tracker_proposal.strobe);
    }
    _prediction.src = ProposalSource::final;
    return _prediction;
}

Candidate TldTracker::GerCurrentPrediction() const {
    return _prediction;
}

Candidate TldTracker::operator<<(const cv::Mat& input_frame) {
    return ProcessFrame(input_frame);
}

void TldTracker::StartTracking(const cv::Rect target) {
    _detector.SetTarget(target);
    _tracker.SetTarget(target);
    _model.SetTarget(target);
    _processing_en = true;
}

void TldTracker::operator <<(cv::Rect target) {
    StartTracking(target);
}

void TldTracker::StopTracking() {
    _processing_en = false;
}

void TldTracker::UpdateSettings() {

}

bool TldTracker::IsProcessing() const {
    return _processing_en;
}

std::tuple<std::vector<Candidate>, std::vector<Candidate>, Candidate> TldTracker::GetProposals() const {
    return make_tuple(_detector_proposals, _integrator.GetClusters(), _tracker_proposal);
}

TldStatus TldTracker::GetStatus() const {
    TldStatus out;
    out.message = _integrator.GetStatusMessage();
    out.training = _training_en;
    out.processing = _processing_en;
    out.valid_object = _prediction.valid;
    out.tracker_relocation = _tracker_relocate;
    out.detector_candidates_cnt = _detector_proposals.size();
    out.detector_clusters_cnt = _integrator.GetClusters().size();
    return out;
}

std::vector<cv::Mat> TldTracker::GetModelsPositive() const {
    return _model.GetPositiveSample();
}

std::vector<cv::Mat> TldTracker::GetModelsNegative() const {
    return _model.GetNegativeSample();
}

}




