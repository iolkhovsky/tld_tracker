#include <tracker/integrator.h>

namespace TLD {

    Integrator::Integrator(ObjectModel& model)
        : _model(model) {
    }

    void Integrator::_preprocess_candidates() {
        _detector_proposal_clusters = clusterize_candidates(_detector_raw_proposals, _settings.clusterization_iou_threshold);

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

        _dc_more_confident_than_tracker.clear();
        std::copy_if(_detector_proposal_clusters.begin(), _detector_proposal_clusters.end(),
                     std::back_inserter(_dc_more_confident_than_tracker), [this] (const Candidate& c) {
            return c.aux_prob > _tracker_raw_proposal.aux_prob;
        });

        _dc_close_to_tracker.clear();
        std::copy_if(_dc_more_confident_than_tracker.begin(), _dc_more_confident_than_tracker.end(),
                     std::back_inserter(_dc_close_to_tracker), [this] (const Candidate& c) {
            return compute_iou(c.strobe, _tracker_raw_proposal.strobe) > 0.75;
        });

        _tracker_raw_proposal.aux_prob = _model.Predict(_tracker_raw_proposal);
    }

    std::tuple<Candidate, bool, bool> Integrator::_get_integration_result() const {
        return std::make_tuple(_final_proposal, _training_enable, _tracker_relocation_enable);
    }

    std::tuple<Candidate, bool, bool> Integrator::Integrate(std::vector<Candidate> det_proposals,
                                    Candidate tracker_proposal) {
        _detector_raw_proposals = det_proposals;
        _tracker_raw_proposal = tracker_proposal;

        _preprocess_candidates();

        bool tracker_result_is_stable = tracker_proposal.valid;
        bool tracker_result_confident = _tracker_raw_proposal.aux_prob > _settings.model_prob_threshold;

        if ( tracker_result_is_stable && tracker_result_confident)
            _subtree_tracker_result_is_reliable();
        else
            _subtree_tracker_result_is_not_reliable();
        return _get_integration_result();
    }

   std::vector<Candidate> Integrator::GetClusters() const {
        return _detector_proposal_clusters;
   }

   void Integrator::SetSettings(IntegratorSettings settings) {
       _settings = settings;
   }

   void Integrator::_subtree_detector_clusters_not_reliable() {
       if (_tracker_raw_proposal.aux_prob > 0.4) {
           _final_proposal.strobe = _tracker_raw_proposal.strobe;
           _final_proposal.src = ProposalSource::tracker;
           _final_proposal.valid = true;
           _final_proposal.prob = _tracker_raw_proposal.aux_prob;
           _training_enable = false;
           _tracker_relocation_enable = false;
           _status_message = "Only tracker's unreliable result";
           if (_tracker_raw_proposal.aux_prob > 0.65) {
              _training_enable = true;
              _tracker_relocation_enable = false;
              _status_message = "Only tracker's reliable result";
           }
       } else {
           _final_proposal.src = ProposalSource::mixed;
           _final_proposal.valid = false;
           _final_proposal.prob = 0.0;
           _training_enable = false;
           _tracker_relocation_enable = false;
           _status_message = "No reliable proposals";
       }
   }

   void Integrator::_subtree_no_more_confident_det_clusters() {
        _subtree_detector_clusters_not_reliable();
   }

   void Integrator::_subtree_one_more_confident_det_cluster() {
        double iou = compute_iou(_dc_more_confident_than_tracker.front().strobe,
                                 _tracker_raw_proposal.strobe);
        if (iou < 0.75) {
            _final_proposal.strobe = _dc_more_confident_than_tracker.front().strobe;
            _final_proposal.src = ProposalSource::detector;
            _final_proposal.valid = true;
            _final_proposal.prob = _dc_more_confident_than_tracker.front().aux_prob;
            _training_enable = false;
            _tracker_relocation_enable = _dc_more_confident_than_tracker.front().aux_prob > 0.65;
            _status_message = "One detectors cluster better than tracker";
        } else {
            std::vector<Candidate> all_candidates;
            all_candidates.push_back(_dc_more_confident_than_tracker.front());
            all_candidates.push_back(_tracker_raw_proposal);
            _final_proposal = aggregate_candidates(all_candidates);
            _final_proposal.src = ProposalSource::mixed;
            _final_proposal.valid = true;
            _final_proposal.prob = _final_proposal.aux_prob;
            _training_enable = true;
            _tracker_relocation_enable = false;
            _status_message = "One Detector & Tracker are close";
        }
   }

   void Integrator::_subtree_few_more_confident_det_clusters() {
        if (_dc_close_to_tracker.empty()) {
            _final_proposal.src = ProposalSource::mixed;
            _final_proposal.valid = false;
            _final_proposal.prob = 0.0;
            _training_enable = false;
            _tracker_relocation_enable = false;
            _status_message = "Few detectors clusters far from tracker";
        } else {
            std::vector<Candidate> all_candidates(_dc_close_to_tracker.begin(),
                                                  _dc_close_to_tracker.end());
            all_candidates.push_back(_tracker_raw_proposal);
            _final_proposal = aggregate_candidates(all_candidates);
            _final_proposal.src = ProposalSource::mixed;
            _final_proposal.valid = true;
            _final_proposal.prob = _final_proposal.aux_prob;
            _training_enable = true;
            _tracker_relocation_enable = false;
            _status_message = "Few Detector & Tracker are close";
        }
   }

   void Integrator::_subtree_tracker_result_is_reliable() {
       if (_detector_proposal_clusters.empty()) {
           _subtree_detector_clusters_not_reliable();
       } else {
           if (_dc_more_confident_than_tracker.empty()) {
               _subtree_no_more_confident_det_clusters();
           } else if (_dc_more_confident_than_tracker.size() == 1)
               _subtree_one_more_confident_det_cluster();
           else
               _subtree_few_more_confident_det_clusters();
        }
   }

   void Integrator::_subtree_tracker_result_is_not_reliable() {
        if (_detector_proposal_clusters.empty()) {
            _final_proposal.src = ProposalSource::mixed;
            _final_proposal.valid = false;
            _final_proposal.prob = 0.0;
            _training_enable = false;
            _tracker_relocation_enable = false;
            _status_message = "No reliable results";
       } else if (_detector_proposal_clusters.size() == 1) {
            bool detector_stable = ((_detector_proposal_clusters.front().prob > 0.7)
                                    || (_detector_proposal_clusters.front().aux_prob > 0.8));
            _final_proposal.strobe = _detector_proposal_clusters.front().strobe;
            _final_proposal.prob = _detector_proposal_clusters.front().aux_prob;
            _final_proposal.valid = true;
            _final_proposal.src = ProposalSource::detector;
            _training_enable = true;
            _tracker_relocation_enable = detector_stable;
            _status_message = "One detector cluster";
       } else {
            if (_detector_proposal_clusters.front().aux_prob > 0.95) {
                _final_proposal.src = ProposalSource::detector;
                _final_proposal.valid = true;
                _final_proposal.prob = _detector_proposal_clusters.front().aux_prob;
                _training_enable = false;
                _tracker_relocation_enable = false;
                _status_message = "Most confident Detector cluster";
            } else {
                _final_proposal.src = ProposalSource::mixed;
                _final_proposal.valid = false;
                _final_proposal.prob = 0.0;
                _training_enable = false;
                _tracker_relocation_enable = false;
                _status_message = "No reliable results";
            }
       }
   }

   std::string Integrator::GetStatusMessage() const {
       return _status_message;
   }

}
