#pragma once

struct implicit_CIR {
    using result_type = double;
    operator result_type() const { return state; }
    template <typename TAlgo, typename TRandom> friend struct random_scheme;

    implicit_CIR() = default;
    implicit_CIR(CIR const & sde, double h = 1)
        : sde(sde), state(sde.init_value), h(h) {}

    template <typename TWhiteNoise>
        result_type operator()(TWhiteNoise const & z) {
            double a = (1 + 0.5* sde.lambda *h);
            double b = - (0.5 * sde.sig * sqrt(h) * z + sqrt(state));
            double c = - 0.5 * (sde.alpha - 0.25 * sde.sig * sde.sig) * h;
            double y = (-b + sqrt(b*b - 4*a*c)) / (2*a);
            return state = y*y;
        }
protected:
    CIR sde;
    result_type state;
    double h;
};
