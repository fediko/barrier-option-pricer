#pragma once

#include <armadillo>

template <int d>
using mat = arma::mat::fixed<d, d>;

template <int d>
using vec = arma::vec::fixed<d>;

using vecMC = std::vector<double>;
void affiche(std::string const & s, vecMC const & v) {
    //std::cout << setiosflags(std::ios_base::fixed)
    std::cout << s << ": \t" << v[0] << "\t" << v[1] << "\t" << v[2] << std::endl;
};

template <int d>
struct normal_dist_md {
    normal_dist_md() = default;

    normal_dist_md(mat<d> const & L) : rac_covar(L), G(0,1) {};

    template <typename TGen>
    vec<d> operator()(TGen & gen) {
        vec<d> x, y;
        for (auto & xk : x) { xk = G(gen); }
        y = rac_covar * x;
        return y;
    };

    private:
        mat<d> rac_covar;
        std::normal_distribution<double> G;
};

template <int d>
struct bernoulli_dist_md { // 1 or -1 with proba p
    bernoulli_dist_md() = default;

    bernoulli_dist_md(mat<d> const & L, double p = 0.5) : rac_covar(L), B(p) {};

    template <typename TGen>
    vec<d> operator()(TGen & gen) {
        vec<d> x;
        for (auto & xk : x) { xk = B(gen) ? 1 : -1; }
        return rac_covar * x;
    };

    private:
        mat<d> rac_covar;
        std::bernoulli_distribution B;
        double p;
};


template <int d>
struct sde_md {
    sde_md() = default;

    sde_md(vec<d> const & x0, std::function<vec<d>(vec<d>)> b,
                     std::function<vec<d>()> s)
        : x0(x0), b(b), sigma(s) {};


    vec<d> x0;
    std::function<vec<d>(vec<d>)> b; // not time dependent
    std::function<vec<d>()> sigma; // sigma is a constant quantity in this model


};

template <int d>
struct black_scholes_md {
    black_scholes_md() = default;

    black_scholes_md(vec<d> const & x0, double r,
                     vec<d> const & s, double T)
        : x0(x0), mu(r - 0.5 * s % s), sigma(s), T(T) {};

    vec<d> operator()(vec<d> const & g) const {
        vec<d> result = x0 % exp(mu * T + sqrt(T) * sigma % g);
        return result;
    };

    private:
        vec<d> x0, mu, sigma;
        double T;
};

template <int d>
struct basket {
    basket() = default;

    basket(vec<d> const & alpha, double K)
        : alpha(alpha), K(K) {};

    double operator()(vec<d> const & x) const {
        double y = dot(x, alpha);
        return (y > K) ? (y - K) : 0;
    };

    private:
        vec<d> alpha;
        double K;
};

template <int d>
struct basket_cv {
    basket_cv() = default;

    basket_cv(vec<d> const & alpha_x, vec<d> const & x0, double K, double I0, double price)
        : alpha_x(alpha_x), x0(x0), K(K), I0(I0), price(price) {};

    double operator()(vec<d> const & x) const {
        double y = I0 * exp(dot(alpha_x, log(x / x0)) / I0);
        return (y > K) ? ((y - K) - price) : - price;
    };

    private:
        vec<d> alpha_x, x0;
        double K, I0, price;
};

template <int d>
struct tneg {
    vec<d> operator()(vec<d> x) const { return -x; };
};
