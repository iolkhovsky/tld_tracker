#pragma once

#include <tuple>
#include <vector>

#include "common.h"
#include "opt_flow_tracker.h"
#include "object_detector.h"
#include "object_model.h"
#include "integrator.h"

namespace tld {

    struct TldStatus {
        bool processing;
        bool valid_object;
        bool training;
        bool tracker_relocation;
        size_t detector_candidates_cnt;
        size_t detector_clusters_cnt;
        std::string message;
    };

    class TldTracker {
    public:
        TldTracker(Settings settings);
        Candidate ProcessFrame(const cv::Mat& input_frame);
        void StartTracking(const cv::Rect target);
        void StopTracking();
        void UpdateSettings();
        bool IsProcessing() const;
        std::tuple<std::vector<Candidate>, std::vector<Candidate>, Candidate> GetProposals() const;
        TldStatus GetStatus() const;
        Candidate GerCurrentPrediction() const;
        Candidate operator<<(const cv::Mat& input_frame);
        void operator <<(cv::Rect target);
        std::vector<cv::Mat> GetModelsPositive() const;
        std::vector<cv::Mat> GetModelsNegative() const;

    private:
        Settings _settings;
        OptFlowTracker _tracker;
        ObjectDetector _detector;
        ObjectModel _model;
        Integrator _integrator;

        cv::Mat _src_frame;
        cv::Mat _prev_frame;
        cv::Mat _lf_frame;

        bool _processing_en;
        bool _training_en;
        bool _tracker_relocate;
        Candidate _prediction;

        std::vector<Candidate> _detector_proposals;
        Candidate _tracker_proposal;
    };

    TldTracker make_tld_tracker();
    TldTracker make_tld_tracker(Settings s);

}

std::ostream& operator<<(std::ostream &os, const tld::TldTracker& tracker);
