#pragma once

#include "object_model.h"
#include "tld_utils.h"

namespace tld {

    class Integrator {
    public:
        Integrator(ObjectModel& model);
        std::tuple<Candidate, bool, bool> Integrate(std::vector<Candidate> det_proposals,
                            Candidate tracker_proposal);
        std::vector<Candidate> GetClusters() const;
        void SetSettings(IntegratorSettings settings);
        std::string GetStatusMessage() const;
    private:
        const ObjectModel& _model;
        IntegratorSettings _settings;
        std::vector<Candidate> _detector_raw_proposals;
        Candidate _tracker_raw_proposal;

        std::vector<Candidate> _detector_proposal_clusters;
        std::vector<Candidate> _dc_more_confident_than_tracker;
        std::vector<Candidate> _dc_close_to_tracker;

        std::string _status_message;
        Candidate _final_proposal;
        bool _training_enable;
        bool _tracker_relocation_enable;

        void _preprocess_candidates();
        std::tuple<Candidate, bool, bool> _get_integration_result() const;


        void _subtree_tracker_result_is_reliable();
        void _subtree_detector_clusters_not_reliable();
        void _subtree_no_more_confident_det_clusters();
        void _subtree_one_more_confident_det_cluster();
        void _subtree_few_more_confident_det_clusters();

        void _subtree_tracker_result_is_not_reliable();
    };

}
