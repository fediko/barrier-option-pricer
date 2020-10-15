#pragma once
#include <functional>

template <typename TState = double, typename TSigma = double>
struct sde {
    using state_type = TState;
    std::function<TState(TState const &)> b;
    std::function<TSigma(TState const &)> sigma;
    TState init_value;
};

struct BS {
    BS(double r = 0, double sigma = 0, double x0 = 1)
        : r(r), sig(sigma), init_value(x0) { }
    using state_type = double;
    double b(double x) const { return r * x; }
    double sigma(double x) const { return sig * x; }
    double r, sig, init_value;
};

struct CIR {
    CIR(double alpha = 0, double lambda = 0, double sigma = 0, double x0 = 1)
        : alpha(alpha), lambda(lambda), sig(sigma), init_value(x0) { }
    using state_type = double;
    double b(double x) const { return alpha - lambda * x; }
    double sigma(double x) const { return sig * std::sqrt(x); }
    double alpha, lambda, sig, init_value;
};
