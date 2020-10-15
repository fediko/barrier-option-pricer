#pragma once

struct exact_bs {
    using result_type = double;
    operator result_type() const { return state; }
    template <typename TAlgo, typename TRandom> friend struct random_scheme;

    exact_bs() = default;
    exact_bs(double r, double sigma, double x0, double h)
        : r(r), sigma(sigma), state(x0), h(h) {}

    template <typename TWhiteNoise>
    double operator()(TWhiteNoise const & z) {
        return state *= exp((r - 0.5*sigma*sigma) * h + sigma * sqrt(h) * z);
    }
protected:
    double r, sigma;
    double state;
    double h;
};
