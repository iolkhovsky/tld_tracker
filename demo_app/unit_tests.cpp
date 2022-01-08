#include "unit_tests.h"

#include <opencv2/opencv.hpp>

#include "tld/tld_tracker.h"

#include "test_runner.h"
#include "profile.h"


void TestFernFeatureExtractor() {
    cv::Mat gray = tld::generate_random_image();
    cv::Mat filtered_frame;
    cv::blur(gray, filtered_frame, cv::Size(7,7));

    cv::Rect designation;
    designation.x = 100;
    designation.y = 150;
    designation.width = 233;
    designation.height = 172;

    cv::Size imsz(gray.cols, gray.rows);
    auto grid = std::make_shared<tld::ScanningGrid>(imsz);
    tld::FernFeatureExtractor fext(grid);
    auto scales = grid->GetScales();

    std::vector<cv::Size> positions_per_scale = grid->GetPositionsCnt();
    size_t scale_id = 0;
    for (auto positions: positions_per_scale) {
        double abs_scale = scales.at(scale_id);
        for (auto y_i = 0; y_i < positions.height; y_i++) {
            for (auto x_i = 0; x_i < positions.width; x_i++) {

                cv::Rect strobe;
                strobe.x = x_i * static_cast<int>(abs_scale * grid->GetOverlap().width);
                strobe.y = y_i * static_cast<int>(abs_scale * grid->GetOverlap().height);
                strobe.width = static_cast<int>(abs_scale * designation.width);
                strobe.height = static_cast<int>(abs_scale * designation.height);

                auto subframe = filtered_frame(strobe);

                auto desc_0 = fext(filtered_frame, {x_i, y_i}, scale_id);
                auto desc_1 = fext.GetDescriptor(subframe);
                auto desc_2 =fext.GetDescriptor(gray, strobe);

                ASSERT_EQUAL(desc_0, desc_1)
                ASSERT_EQUAL(desc_1, desc_2)
                ASSERT_EQUAL(desc_0, desc_2)
            }
        }
    }
}

void TestObjectClassifier() {
    tld::ObjectClassifier<uint8_t, 8> clf;

    clf.TrainPositive(2);
    clf.TrainPositive(2);
    clf.TrainPositive(2);
    clf.TrainNegative(2);
    ASSERT_EQUAL_EPS(clf.Predict(2), 0.75, std::numeric_limits<double>::epsilon());

    clf.TrainPositive(3);
    clf.TrainPositive(3);
    clf.TrainNegative(3);
    ASSERT_EQUAL_EPS(clf.Predict(3), 0.66667, 0.01);

    clf.TrainPositive(7);
    clf.TrainNegative(7);
    ASSERT_EQUAL_EPS(clf.Predict(7), 0.5, std::numeric_limits<double>::epsilon());

    clf.TrainPositive(6);
    ASSERT_EQUAL_EPS(clf.Predict(6), 1.0, std::numeric_limits<double>::epsilon());

    ASSERT_EQUAL_EPS(clf.Predict(0), 0.0, std::numeric_limits<double>::epsilon());

    ASSERT_EQUAL(clf.GetMaxPositive(), 3);
    ASSERT_EQUAL(clf.GetPositiveDistr(7), 1);
    ASSERT_EQUAL(clf.GetNegativeDistr(0), 0);

    clf.Reset();
    ASSERT_EQUAL(clf.GetPositiveDistr(3), 0);
}

void TestAugmentator() {
    cv::Mat frame = tld::generate_random_image({100, 100});
    cv::Rect strobe(10,20,30,40);
    tld::TranformPars pars;

    pars.angles = {-10, 24, 12};
    pars.disp_threshold = 0.1;
    pars.neg_sample_size_limit = -1;
    pars.overlap = 0.05;
    pars.pos_sample_size_limit = -1;
    pars.scales = {0.4, 0.7};
    pars.translation_x = {0};
    pars.translation_y = {0};

    tld::Augmentator aug(frame, strobe, pars);

    int cnt = 0;
    for (auto img: aug.SetClass(tld::ObjectClass::Positive))
        cnt++;
    ASSERT_EQUAL(cnt, 6);

    pars.pos_sample_size_limit = 4;
    tld::Augmentator aug2(frame, strobe, pars);

    cnt = 0;
    for (auto img: aug2.SetClass(tld::ObjectClass::Positive))
        cnt++;
    ASSERT_EQUAL(cnt, 4);
}

void TestObjectModel() {
    tld::ObjectModel model;
    cv::Mat frame = tld::generate_random_image({200, 200});
    auto fptr = std::make_shared<cv::Mat>(frame);
    model.SetFrame(fptr);
    cv::Rect ref_strobe(10, 20, 40, 50);
    model.SetTarget(ref_strobe);;
    double prob = model.Predict(frame(ref_strobe));
    ASSERT_EQUAL_EPS(prob, 1.0, 0.05);
}

void TestOptFlow() {
    tld::OptFlowTracker tracker;
    cv::Mat prev_frame(640, 480, CV_8UC1);
    cv::Mat cur_frame(640, 480, CV_8UC1);
    cv::Size offset(15, 7);

    prev_frame.setTo(0);
    cv::rectangle(prev_frame, {100, 120, 100, 150}, cv::Scalar(255), -1);
    tracker.SetFrame(prev_frame);
    cv::Rect target(80, 100, 150, 200);
    tracker.SetTarget(target);

    cur_frame.setTo(0);
    cv::rectangle(cur_frame, {100 + offset.width, 120 + offset.height, 100, 150}, cv::Scalar(255), -1);
    tracker.SetFrame(cur_frame);

    auto res = tracker.Track();

    ASSERT_EQUAL_EPS(res.strobe.x, target.x + offset.width, 1);
    ASSERT_EQUAL_EPS(res.strobe.y, target.y + offset.height, 1);
}

void TestUtils() {
    cv::Rect a(10,10,10,10);
    cv::Rect b(15,15,10,10);
    double iou = tld::compute_iou(a, b);
    ASSERT_EQUAL_EPS(iou, 1./7., std::numeric_limits<double>::epsilon());
}

void run_tests() {
    TestRunner tr;
    RUN_TEST(tr, TestFernFeatureExtractor);
    RUN_TEST(tr, TestObjectClassifier);
    RUN_TEST(tr, TestAugmentator);
    RUN_TEST(tr, TestObjectModel);
    RUN_TEST(tr, TestOptFlow);
    RUN_TEST(tr, TestUtils);
}
