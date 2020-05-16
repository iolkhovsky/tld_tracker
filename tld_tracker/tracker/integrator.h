#pragma once

#include <tracker/object_model.h>
#include <tracker/tld_utils.h>

namespace TLD {

    class Integrator {
    public:
        Integrator(ObjectModel& model);
        std::tuple<Candidate, bool> Integrate(std::vector<Candidate> det_proposals,
                            Candidate tracker_proposal);
        std::vector<Candidate> GetClusters() const;
        void SetSettings(IntegratorSettings settings);
    private:
        std::vector<Candidate> _detector_proposal_clusters;
        const ObjectModel& _model;
        IntegratorSettings _settings;
        std::string _status_message;
        Candidate _result;
    };

}
