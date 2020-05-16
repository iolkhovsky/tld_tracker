#include <tracker/integrator.h>

namespace TLD {

    Integrator::Integrator(ObjectModel& model)
        : _model(model) {
    }

    std::tuple<Candidate, bool> Integrator::Integrate(std::vector<Candidate> det_proposals,
                                    Candidate tracker_proposal) {
        _detector_proposal_clusters = clusterize_candidates(det_proposals, _settings.clusterization_iou_threshold);
        auto validate_by_model = [this] (Candidate& item) {
            item.aux_prob = _model.Predict(item);
            return item.aux_prob < _settings.model_prob_threshold;
        };
        _detector_proposal_clusters.erase(
                    std::remove_if(_detector_proposal_clusters.begin(), _detector_proposal_clusters.end(), validate_by_model),
                    _detector_proposal_clusters.end());
        std::sort(_detector_proposal_clusters.begin(), _detector_proposal_clusters.end(), [] (const Candidate& lhs, const Candidate& rhs) {
            return lhs.aux_prob > rhs.aux_prob;
        });
        tracker_proposal.aux_prob = _model.Predict(tracker_proposal);


        if ((tracker_proposal.aux_prob > _settings.model_prob_threshold)
                && (tracker_proposal.prob > _settings.tracker_prob_threshold)) {
            // tracker's result is stable

            if (_detector_proposal_clusters.empty()) {
                // no detector's proposals
                if (tracker_proposal.aux_prob > 0.4) {
                    _result.strobe = tracker_proposal.strobe;
                    _result.src = ProposalSource::tracker;
                    _result.valid = true;
                    _result.prob = tracker_proposal.aux_prob;
                    _result.training = false;
                    _status_message = "Tracker's one unstable strobe";
                    if (tracker_proposal.aux_prob > 0.65) {
                       _result.training = true;
                       _status_message = "Tracker's one stable strobe";
                    }
                }
            } else {
                std::vector<Candidate> clusters_better_than_tracker;
                std::vector<Candidate> close_clusters_better_than_tracker;
                std::copy_if(_detector_proposal_clusters.begin(), _detector_proposal_clusters.end(),
                             std::back_inserter(clusters_better_than_tracker), [tracker_proposal] (const Candidate& c) {
                    return c.aux_prob > tracker_proposal.aux_prob;
                });
                std::copy_if(clusters_better_than_tracker.begin(), clusters_better_than_tracker.end(),
                             std::back_inserter(close_clusters_better_than_tracker), [tracker_proposal] (const Candidate& c) {
                    return compute_iou(c.strobe, tracker_proposal.strobe) > 0.75;
                });


                if (clusters_better_than_tracker.empty()) {
                    _result.strobe = tracker_proposal.strobe;
                    _result.src = ProposalSource::tracker;
                    _result.valid = true;
                    _result.prob = tracker_proposal.aux_prob;
                    _result.training = false;
                    _status_message = "Tracker's one unstable strobe";
                    if (tracker_proposal.aux_prob > 0.65) {
                       _result.training = true;
                       _status_message = "Tracker's one stable strobe";
                    }
                } else if (clusters_better_than_tracker.empty()==1) {
                    double iou = compute_iou(clusters_better_than_tracker.back().strobe, tracker_proposal.strobe);
                    if (iou < 0.75) {
                        _result.strobe = clusters_better_than_tracker.back().strobe;
                        _result.src = ProposalSource::detector;
                        _result.valid = true;
                        _result.prob = clusters_better_than_tracker.back().aux_prob;
                        _result.training = false;
                        _status_message = "Detector's one cluster";
                    } else {
                        _result.strobe = clusters_better_than_tracker.back().strobe;
                        _result.src = ProposalSource::mixed;
                        _result.valid = true;
                        _result.prob = clusters_better_than_tracker.back().aux_prob;
                        _result.training = true;
                        _status_message = "Detector & Tracker one common proposal";
                    }
                } else {
                    if (close_clusters_better_than_tracker.empty()) {
                        _result.strobe = tracker_proposal.strobe;
                        _result.src = ProposalSource::mixed;
                        _result.valid = false;
                        _result.prob = 0.0;
                        _result.training = false;
                        _status_message = "All good clusters to far from tracker";
                    } else {
                        // average strobe
                        std::vector<Candidate> all(close_clusters_better_than_tracker.begin(),
                                                   close_clusters_better_than_tracker.end());
                        all.push_back(tracker_proposal);
                        _result = aggregate_candidates(all);
                        _result.prob = _result.aux_prob;
                        _result.src = ProposalSource::mixed;
                        _result.valid = true;
                        _result.training = true;
                        _status_message = "Tracker & Detector with few clusters";
                    }
                }
            }
        } else {
            if (_detector_proposal_clusters.empty()) {
                _result.src = ProposalSource::mixed;
                _result.prob = 0.0;
                _result.valid = false;
                _result.training = false;
                _status_message = "Unstable Tracker and Detector";
            } else if (_detector_proposal_clusters.size()==1) {
                bool stable = (_detector_proposal_clusters.back().prob > 0.8)
                        && (_detector_proposal_clusters.back().aux_prob > 0.75);
                _result.strobe = _detector_proposal_clusters.back().strobe;
                _result.src = ProposalSource::detector;
                _result.prob = _detector_proposal_clusters.back().aux_prob;
                _result.valid = true;
                _result.training = stable;
                _status_message = "Detector one stable clusters";
            } else {
                //2.2.1. Если валидность кластера с макс. значением очень высокая
                if (_detector_proposal_clusters.front().aux_prob > 0.95) {
                    _result.strobe = _detector_proposal_clusters.front().strobe;
                    _result.src = ProposalSource::detector;
                    _result.prob = _detector_proposal_clusters.front().aux_prob;
                    _result.valid = true;
                    _result.training = false;
                    _status_message = "Most prob Detectors result";
                } else {
                    _result.src = ProposalSource::mixed;
                    _result.prob = 0.0;
                    _result.valid = false;
                    _result.training = false;
                    _status_message = "Unstable Tracker and Detector";
                }
            }
        }
        return {_result, _result.training};
    }

   std::vector<Candidate> Integrator::GetClusters() const {
        return _detector_proposal_clusters;
   }

   void Integrator::SetSettings(IntegratorSettings settings) {
       _settings = settings;
   }

}
