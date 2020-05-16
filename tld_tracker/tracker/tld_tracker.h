#pragma once

#include <tuple>
#include <vector>

#include <tracker/common.h>
#include <tracker/opt_flow_tracker.h>
#include <tracker/object_detector.h>
#include <tracker/object_model.h>
#include <tracker/integrator.h>

namespace TLD {

    class TldTracker {
    public:
        TldTracker(Settings settings);
        Candidate SetFrame(const cv::Mat& input_frame);
        void StartTracking(const cv::Rect target);
        void StopTracking();
        void UpdateSettings();
        bool IsProcessing();
        std::tuple<std::vector<Candidate>, Candidate> GetProposals();

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
        Candidate _prediction;

        std::vector<Candidate> _detector_proposals;
        Candidate _tracker_proposal;
    };

}

