#pragma once
#include <iostream>
#include <vector>
#include <thread>

struct mean_var {
    mean_var(unsigned n = 0, double sum_x = 0, double sum_xx = 0)
        : sample_size(n), sum_x(sum_x), sum_xx(sum_xx) { }
    double mean() const { return sum_x / (double) sample_size; }
    double var() const { return (sum_xx - sample_size * mean() * mean())
                 / (double) (sample_size-1); }
    double ic_size() const { return 1.96 * std::sqrt(var() / sample_size); }

    mean_var & operator+=(mean_var const & mv) {
        sample_size += mv.sample_size;
        sum_x += mv.sum_x;
        sum_xx += mv.sum_xx;
        return *this;
    }
    friend mean_var operator+(mean_var const & mv1, mean_var const & mv2) {
        return { mv1.sample_size + mv2.sample_size,
                 mv1.sum_x + mv2.sum_x,
                 mv1.sum_xx + mv2.sum_xx };
    }
    friend mean_var operator*(double alpha, mean_var const & mv) {
        return { mv.sample_size, alpha * mv.sum_x, alpha * alpha * mv.sum_xx };
    }
    friend std::ostream & operator<<(std::ostream & o, mean_var const & mv) {
        return o << "Size: " << mv.sample_size
         << "\tMean: " << mv.mean()
         << "\tVar: " << mv.var()
         << "\tCI size: " << mv.ic_size();
    }
    protected:
        unsigned sample_size;
        double sum_x, sum_xx;
};

template <typename TDistrib, typename TGen>
mean_var monte_carlo(TDistrib & X, TGen & gen, unsigned sample_size) {
    double sum_x = 0, sum_xx = 0;
    for (unsigned k = 0; k < sample_size; ++k) {
        double x = X(gen);
        sum_x += x;
        sum_xx += x*x;
    }
    return { sample_size, sum_x, sum_xx };
}

template <typename TDistrib, typename TGen>
mean_var monte_carlo(TDistrib & X, TGen & gen, unsigned batch_size, double epsilon) {
    auto r = monte_carlo(X, gen, batch_size);
    while (r.ic_size() > epsilon) {
        auto tmp = monte_carlo(X, gen, batch_size);
        r += tmp;
    }
    return r;
}

/*** on desactive la version OpenMP ***/
/* template <typename TDistrib, typename TGen> */
/* mean_var monte_carlo_openmp(TDistrib & X, TGen & gen, unsigned sample_size) { */
/*     double sum_x = 0, sum_xx = 0; */
/*     #pragma omp parallel for */
/*     for (unsigned k = 0; k < sample_size; ++k) { */
/*      double x = X(gen); */
/*         sum_x += x; */
/*         sum_xx += x*x; */
/*     } */
/*     return { sample_size, sum_x, sum_xx }; */
/* } */

template <typename TDistrib, typename TGen>
mean_var monte_carlo_multithread(TDistrib & X, std::vector<TGen> & gens, unsigned sample_size) {
    unsigned num_thread = gens.size();
    std::vector<std::thread> threads(num_thread);
    std::vector<mean_var> mvs(num_thread);
    unsigned q = sample_size / num_thread;
    unsigned r = sample_size % num_thread;
    for (int k = 0; k < num_thread; ++k) {
        unsigned M_k = q + (k < r ? 1 : 0);
        auto & gen_k = gens[k];
        auto & mv_k = mvs[k];
        threads[k] = std::thread([X, &gen_k, M_k, &mv_k]() mutable {
                mv_k = monte_carlo(X, gen_k, M_k);
            });
    }
    for (auto & th : threads) th.join();
    mean_var result;
    for (auto const & mv : mvs) result += mv;
    return result;
}

template <typename TDistrib, typename TGen>
mean_var stratification_prop(std::vector<TDistrib> X, TGen & gen,
               unsigned sample_size, std::vector<double> probas) {
    double mean = 0, var = 0;
for (unsigned k = 0; k < X.size(); ++k) {
        unsigned Mk = std::ceil(sample_size * probas[k]);
    auto mv_k = monte_carlo(X[k], gen, Mk);
        mean += probas[k] * mv_k.mean();
    var += probas[k] * mv_k.var();
  }
  double sum_x = sample_size * mean;
  double sum_xx = (sample_size-1) * var + sum_x * mean;
    return { sample_size, sum_x, sum_xx };
}

template <typename TFun, typename Trans>
struct antithetic_t
{
    antithetic_t() = default;
    antithetic_t(TFun f, Trans t_anti) : f(f), t_anti(t_anti) {};
    template <typename T>
    auto operator()(T && x) -> decltype(TFun()(x)) {
        return 0.5*(f(x) + f(t_anti(x)));
    }
private:
    TFun f;
    Trans t_anti;
};

template <typename TFun, typename Trans>
inline antithetic_t<TFun, Trans>
antithetic(TFun f, Trans t_anti) {
    return antithetic_t<TFun, Trans>(f, t_anti);
};

template <typename TFun1, typename TFun2>
struct control_variate_t
{
    control_variate_t() = default;
    control_variate_t(TFun1 f1, TFun2 f2) : f1(f1), f2(f2) {};
    template <typename T>
    auto operator()(T && x) -> decltype(TFun1()(x)) {
        return f1(x) - f2(x);
    }
private:
    TFun1 f1;
    TFun2 f2;
};

template <typename TFun1, typename TFun2>
struct control_variate_adapt_t
{
    control_variate_adapt_t() = default;
    control_variate_adapt_t(TFun1 f1, TFun2 f2, double lambda)
        : f1(f1), f2(f2), cov(0), var(0), lambda(lambda) {};
    template <typename T>
    auto operator()(T && x) -> decltype(TFun1()(x)) {
        double f1_x = f1(x), f2_x = f2(x);
        double result = f1_x - lambda*f2_x;
        cov += f1_x * f2_x;
        var += f2_x * f2_x;
        lambda = var > 0 ? cov / var : lambda;
        return result;
    }
    double get_lambda() const { return lambda; }
private:
    TFun1 f1;
    TFun2 f2;
    double cov, var, lambda;
};

template <typename TFun1, typename TFun2>
inline control_variate_t<TFun1, TFun2>
control_variate(TFun1 f1, TFun2 f2) {
    return control_variate_t<TFun1, TFun2>(f1, f2);
};

template <typename TFun1, typename TFun2>
inline control_variate_adapt_t<TFun1, TFun2>
control_variate(TFun1 f1, TFun2 f2, double & lambda) {
    return control_variate_adapt_t<TFun1, TFun2>(f1, f2, lambda);
};
