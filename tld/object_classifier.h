#pragma once

#include <vector>

namespace tld {

    template<typename descriptor, size_t descriptors_cnt>
    class ObjectClassifier {
    public:
        ObjectClassifier();
        void Reset();
        size_t TrainPositive(descriptor x);
        size_t TrainNegative(descriptor x);
        size_t GetMaxPositive();
        double Predict(descriptor x);
        size_t GetPositiveDistr(descriptor x);
        size_t GetNegativeDistr(descriptor x);
    private:
        std::vector<size_t> _positive_distribution;
        std::vector<size_t> _negative_distribution;
        std::vector<double> _posterior_prob_distribution;
        std::vector<bool> _updated;
        size_t _positive_distr_max;

        double _update_prob(descriptor x);
    };

    template<typename descriptor, size_t descriptors_cnt>
    ObjectClassifier<descriptor, descriptors_cnt>::ObjectClassifier() :
        _positive_distribution(descriptors_cnt, 0),
        _negative_distribution(descriptors_cnt, 0),
        _posterior_prob_distribution(descriptors_cnt, 0.0),
        _updated(descriptors_cnt, false),
        _positive_distr_max(0) {
    }

    template<typename descriptor, size_t descriptors_cnt>
    void ObjectClassifier<descriptor, descriptors_cnt>::Reset() {
        _positive_distribution.assign(descriptors_cnt, 0);
        _negative_distribution.assign(descriptors_cnt, 0);
        _posterior_prob_distribution.assign(descriptors_cnt, 0.0);
        _updated.assign(descriptors_cnt, false);
        _positive_distr_max = 0;
    }
    template<typename descriptor, size_t descriptors_cnt>
    size_t ObjectClassifier<descriptor, descriptors_cnt>::TrainPositive(descriptor x) {
        _updated[x] = false;
        size_t res = ++_positive_distribution[x];
        _positive_distr_max = std::max(_positive_distr_max, res);
        return res;
    }

    template<typename descriptor, size_t descriptors_cnt>
    size_t ObjectClassifier<descriptor, descriptors_cnt>::TrainNegative(descriptor x) {
        _updated[x] = false;
        return ++_negative_distribution[x];
    }

    template<typename descriptor, size_t descriptors_cnt>
    size_t ObjectClassifier<descriptor, descriptors_cnt>::GetMaxPositive() {
        return _positive_distr_max;
    }

    template<typename descriptor, size_t descriptors_cnt>
    double ObjectClassifier<descriptor, descriptors_cnt>::Predict(descriptor x) {
        if (_updated[x])
            return _posterior_prob_distribution[x];
        else
            return _update_prob(x);
    }

    template<typename descriptor, size_t descriptors_cnt>
    size_t ObjectClassifier<descriptor, descriptors_cnt>::GetPositiveDistr(descriptor x) {
        return _positive_distribution[x];
    }

    template<typename descriptor, size_t descriptors_cnt>
    size_t ObjectClassifier<descriptor, descriptors_cnt>::GetNegativeDistr(descriptor x) {
        return _negative_distribution[x];
    }

    template<typename descriptor, size_t descriptors_cnt>
    double ObjectClassifier<descriptor, descriptors_cnt>::_update_prob(descriptor x) {
        auto p_cnt = _positive_distribution[x];
        auto n_cnt = _negative_distribution[x];
        if (p_cnt == 0) {
            _posterior_prob_distribution[x] =  0.0;
        } else if (n_cnt != 0){
            _posterior_prob_distribution[x] =
                    static_cast<double>(p_cnt) / (p_cnt + n_cnt);
        } else {
            _posterior_prob_distribution[x] = 1.0;
        }
        _updated[x] = true;
        return _posterior_prob_distribution[x];
    }

}
