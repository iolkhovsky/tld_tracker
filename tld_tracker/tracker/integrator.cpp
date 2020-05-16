#include <tracker/integrator.h>

namespace TLD {

    Integrator::Integrator(ObjectModel& model) {

    }
    std::tuple<Candidate, bool> Integrator::Integrate(std::vector<Candidate> det_proposals,
                                    Candidate tracker_proposal) {
        _detector_proposal_clusters = clusterize_candidates(det_proposals, 0.5);
        return {tracker_proposal, false};
    }

   std::vector<Candidate> Integrator::GetClusters() const {
        return _detector_proposal_clusters;
   }

}
