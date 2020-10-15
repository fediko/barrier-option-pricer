#include <iostream>
#include <vector>
#include <random>
#include <iterator>
#include "compose.hpp"
#include "mc.hpp"
#include "sde.hpp"
#include "schemes.hpp"

using namespace std;

template <typename TAlgo, typename TRandom>
struct random_walk {
    using result_type = typename TAlgo::result_type;

    random_walk() = default;

    random_walk(TAlgo algo, TRandom Z, unsigned n_max, double barrier)
        : algo(algo), Z(Z),  algo_init(algo), n(0), n_max(n_max), barrier(barrier), knocked(false) { }

    void init() { n = 0; algo = algo_init; knocked = false;}

    bool not_end() { return ((n < n_max) && (!knocked)); }

    bool boundaryCondition(result_type step){
      return (st() >= (barrier - step));
    }

    operator result_type() const { return algo.getState(); }

    result_type st() const { return algo.getState(); }

    template <typename TRandomType>
    random_walk & next(TRandomType const & realization) {
      result_type step = -0.5 * algo.getSde().sigma(st()) * algo.getSde().sigma(st()) * algo.getH()
                             + sqrt(algo.getH()) * algo.getSde().sigma(st());

      if (boundaryCondition(step)){
        random_device rd;
        auto seed = rd();
        mt19937_64 gen(seed);
        bernoulli_distribution G(step / (barrier - st() + step));
        if (G(gen)){
          knocked = true;
        }
        else{
          algo.setState(algo.getState() - step);
          ++n;
          algo(realization);
        }
      }
      else{
          ++n;
          algo(realization);
      }
        return *this;
    }

    template <typename TGen>
    random_walk<TAlgo, TRandom> & operator()(TGen & gen);
    bool getKnocked() const { return knocked;};
public:
    TAlgo algo;
    TRandom Z;
protected:
    TAlgo algo_init;
    unsigned n, n_max;
    double barrier;
    bool knocked;
};

template <typename TAlgo, typename TRandom>
template <typename TGen>
random_walk<TAlgo, TRandom> & //::result_type
random_walk<TAlgo, TRandom>::operator()(TGen & gen) {
    init();
    while (not_end()) {
      if (Z(gen)) next(1);
      else next(-1);
    }
    return *this;
};


template <typename TAlgo, typename TRandom>
random_walk<TAlgo, TRandom>
make_random_walk(TAlgo algo, TRandom Z, unsigned n_max, double barrier) {
    return { algo, Z, n_max, barrier };
};

struct BarrierCap {
    BarrierCap() = default;

    BarrierCap(double K)
        : K(K) {};
    template <typename TAlgo, typename TRandom>
    double operator()(random_walk<TAlgo, TRandom> & algo) const {
      if (algo.getKnocked()) { cout << "knocked path" << endl;return 0;}
      double y = exp(algo.st());
      double payoff = (y > K) ? (y - K) : 0;
      cout << "Path payoff = " << payoff << endl;
      return payoff;
    };

    private:
        double K;
};
