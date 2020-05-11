#pragma once

#include <tracker/object_model.h>
#include <tracker/tld_utils.h>

namespace TLD {

    class Integrator {
    public:
        Integrator(ObjectModel& model);
        std::tuple<Candidate, bool> Integrate(std::vector<Candidate> det_proposals,
                            Candidate tracker_proposal);

    };

}
