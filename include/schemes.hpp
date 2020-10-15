#pragma once
#include <armadillo>
#include <iostream>
using namespace std;
template<int d>
struct sde_md;
template <int d>
using mat = arma::mat::fixed<d, d>;

template <int d>
using vec = arma::vec::fixed<d>;


template <typename TAlgo, typename TRandom>
struct random_scheme {
    using result_type = typename TAlgo::result_type;

    random_scheme() = default;
    random_scheme(TAlgo algo, TRandom Z, unsigned n_max)
        : algo(algo), Z(Z),  algo_init(algo), n(0), n_max(n_max) { }
    void init() { n = 0; algo = algo_init; }
    bool not_end() { return n < n_max; }
    random_scheme refine(int k = 2) {
        auto result = *this;
        result.n_max *= k;
        result.algo_init.h /= (double) k;
        return result;
    }

    operator result_type() const { return algo.state; }
    result_type st() const { return algo.state; }

    template <typename TRandomType>
    random_scheme & next(TRandomType const & realization) {
        ++n;
        algo(realization);
        return *this;
    }

    template <typename TGen>
    random_scheme & operator()(TGen & gen);
public:
    TAlgo algo;
    TRandom Z;
protected:
    TAlgo algo_init;
    unsigned n, n_max;
};

template <typename TAlgo, typename TRandom>
template <typename TGen>
random_scheme<TAlgo, TRandom> & //::result_type
random_scheme<TAlgo, TRandom>::operator()(TGen & gen) {
    init();
    while (not_end()) {
        next(Z(gen));
    }
    return *this;
};


template <typename TAlgo, typename TRandom>
random_scheme<TAlgo, TRandom>
make_random_scheme(TAlgo algo, TRandom Z, unsigned n_max) {
    return { algo, Z, n_max };
    //    return random_scheme<TAlgo, TRandom>(algo, Z, n_max);
};

template <typename TSde, typename TState = typename TSde::state_type>
struct euler {
    using result_type = TState;
    operator result_type() const { return state; }
    template <typename TAlgo, typename TRandom> friend struct random_scheme;

    euler() = default;
    euler(TSde const & sde, double h = 1)
        : sde(sde), state(sde.init_value), h(h) {}
    //result_type st() { return state;}
    template <typename TWhiteNoise>
    result_type operator()(TWhiteNoise const & z) {
        auto diffusive_part = sqrt(h) * sde.sigma(state) * z;
        return state += sde.b(state) * h + diffusive_part;
    }

    result_type getState() const {return state;}

    TSde getSde() const { return sde;}

    double getH() const { return h;}

    void setState(result_type s) { state = s;}
protected:
    TSde sde;
    result_type state;
    double h;
};

template <int d>
struct euler_md {
    using result_type = vec<d>;
    operator result_type() const { return state; }
    template <typename TAlgo, typename TRandom,int > friend struct random_walk_md;

    euler_md() = default;
    euler_md(sde_md<d> const & sde, double h = 1)
        : sde(sde), state(sde.x0), h(h) {}
    result_type st() { return state;}
    void operator()(vec<d> const & z) {
        vec<d> diffusive_part = sqrt(h) * sde.sigma() % z; // sigma is a constant quantity, z = sum Uij * xi_j,k+1
        vec<d> drift_part = (sde.b(state) * h);

        state = state + sde.b(state) * h + diffusive_part;
    }

    vec<d> getState() const {return state;}

    sde_md<d> getSde() const { return sde;}

    double getH() const { return h;}

    void setState(vec<d> s) { state = s;}
protected:
    sde_md<d> sde;
    vec<d> state;
    double h;
};

template <typename TSde, typename TState = double>
struct milstein {
    using result_type = TState;
    operator result_type() const { return state; }
    template <typename TAlgo, typename TRandom> friend struct random_scheme;

    milstein() = default;
    milstein(TSde const & sde, double h)
        : sde(sde), state(sde.init_value), h(h) {}

    result_type operator()(double z) {
        double sigma_tilde = sde.sigma(state + 0.5 * sde.sigma(state) * sqrt(h) * (z-1));
        return state += sde.b(state) * h - sde.sigma(state) * sqrt(h)
            + sigma_tilde * sqrt(h) * (z + 1);
    }
protected:
    TSde sde;
    result_type state;
    double h;
};

template <typename TSde, typename TState = typename TSde::state_type>
struct euler_abs {
    using result_type = TState;
    operator result_type() const { return state; }
    template <typename TAlgo, typename TRandom> friend struct random_scheme;

    euler_abs() = default;
    euler_abs(TSde const & sde, double h = 1)
        : sde(sde), state(sde.init_value), h(h) {}

    template <typename TWhiteNoise>
    result_type operator()(TWhiteNoise const & z) {
        auto diffusive_part = sqrt(h) * sde.sigma(state) * z;
        return state += std::fabs(sde.b(state) * h + diffusive_part);
    }
protected:
    TSde sde;
    result_type state;
    double h;
};

template <typename TSde, typename TState = double>
struct milstein_abs {
    using result_type = TState;
    operator result_type() const { return state; }
    template <typename TAlgo, typename TRandom> friend struct random_scheme;

    milstein_abs() = default;
    milstein_abs(TSde const & sde, double h)
        : sde(sde), state(sde.init_value), h(h) {}

    result_type operator()(double z) {
        double sigma_tilde = sde.sigma(state + 0.5 * sde.sigma(state) * sqrt(h) * (z-1));
        return state += std::fabs(sde.b(state) * h - sde.sigma(state) * sqrt(h)
                + sigma_tilde * sqrt(h) * (z + 1));
    }
protected:
    TSde sde;
    result_type state;
    double h;
};


struct dmax {
    double value;
    double max;
};

template <typename TSde, typename TState = dmax>
struct euler_with_max {
    using result_type = TState;
    operator result_type() const { return state; }
    template <typename TAlgo, typename TRandom> friend struct random_scheme;

    euler_with_max() = default;
    euler_with_max(TSde const & sde, double h)
        : sde(sde), state{ sde.init_value, sde.init_value } , h(h) {}

    template <typename TWhiteNoise>
    result_type operator()(TWhiteNoise const & z) {
        auto diffusive_part = sqrt(h) * sde.sigma(state.value) * z;
        state.value += sde.b(state.value) * h + diffusive_part;
        state.max = std::max(state.value, state.max);
        return state;
    }
protected:
    TSde sde;
    result_type state;
    double h;
};

template <typename TSde, typename TState = dmax>
struct milstein_with_max {
    using result_type = TState;
    operator result_type() const { return state; }
    template <typename TAlgo, typename TRandom> friend struct random_scheme;

    milstein_with_max() = default;
    milstein_with_max(TSde const & sde, double h)
        : sde(sde), state{ sde.init_value, sde.init_value } , h(h) {}

    template <typename TWhiteNoise>
    result_type operator()(TWhiteNoise const & z) {
        double sigma_tilde = sde.sigma(state.value
            + 0.5 * sde.sigma(state.value) * sqrt(h) * (z-1));
        state.value += sde.b(state.value) * h - sde.sigma(state.value) * sqrt(h)
            + sigma_tilde * sqrt(h) * (z + 1);
        state.max = std::max(state.value, state.max);
        return state;
    }
protected:
    TSde sde;
    result_type state;
    double h;
};


template <typename TSde, typename TState = dmax>
struct euler_with_simu_max {
    using result_type = TState;
    operator result_type() const { return state; }
    template <typename TAlgo, typename TRandom> friend struct random_scheme;

    euler_with_simu_max() = default;
    euler_with_simu_max(TSde const & sde, double h)
        : sde(sde), state{ sde.init_value, sde.init_value } , h(h) {}

    template <typename TRandom>
    result_type operator()(TRandom const & z) {
        double diffusive_part = sqrt(h) * sde.sigma(state.value) * z.first;
        double delta = sde.b(state.value) * h + diffusive_part;
        double local_max = state.value + 0.5 * delta
            + sqrt(delta * delta - 2 * sde.sigma(state.value) * h * log(z.second));

        state.value += delta;
        state.max = std::max(local_max, state.max);
        return state;
    }
protected:
    TSde sde;
    result_type state;
    double h;
};


struct dmean {
    double value;
    double mean;
};

template <typename TSde, typename TState = dmean>
struct euler_with_mean {
    using result_type = TState;
    operator result_type() const { return state; }
    template <typename TAlgo, typename TRandom> friend struct random_scheme;

    euler_with_mean() = default;
    euler_with_mean(TSde const & sde, double h)
        : sde(sde), state{ sde.init_value, sde.init_value } , h(h) {}

    template <typename TWhiteNoise>
    result_type operator()(TWhiteNoise const & z) {
        state.mean += state.value * h;
        auto diffusive_part = sqrt(h) * sde.sigma(state.value) * z;
        state.value += sde.b(state.value) * h + diffusive_part;
        return state;
    }
protected:
    TSde sde;
    result_type state;
    double h;
};


template <typename TSde, typename TState = dmean>
struct euler_with_trapmean {
    using result_type = TState;
    operator result_type() const { return state; }
    template <typename TAlgo, typename TRandom> friend struct random_scheme;

    euler_with_trapmean() = default;
    euler_with_trapmean(TSde const & sde, double h)
        : sde(sde), state{ sde.init_value, sde.init_value } , h(h) {}

    template <typename TWhiteNoise>
    result_type operator()(TWhiteNoise const & z) {
        state.mean += 0.5 * state.value * h;
        auto diffusive_part = sqrt(h) * sde.sigma(state.value) * z;
        state.value += sde.b(state.value) * h + diffusive_part;
        state.mean += 0.5 * state.value * h;
        return state;
    }
protected:
    TSde sde;
    result_type state;
    double h;
};

template <typename TSde, typename TState = dmean>
struct euler_with_simumean {
    using result_type = TState;
    operator result_type() const { return state; }
    template <typename TAlgo, typename TRandom> friend struct random_scheme;

    euler_with_simumean() = default;
    euler_with_simumean(TSde const & sde, double h)
        : sde(sde), state{ sde.init_value, sde.init_value } , h(h) {}

    template <typename TRandom>
    result_type operator()(TRandom const & z) {
        auto bk = sde.b(state.value);
        auto sk = sde.sigma(state.value);
        state.mean += state.value * h + (bk * 0.5*h*h) + sk * (0.5*h*sqrt(h) * z.first + (h*h*h/12.) * z.second);
        state.value += bk * h + sqrt(h) * sk * z.first;
        return state;
    }
protected:
    TSde sde;
    result_type state;
    double h;
};
