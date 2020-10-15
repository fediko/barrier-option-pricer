#include <iostream>
#include <vector>
#include <random>
#include <iterator>
#include "compose.hpp"
#include "mc.hpp"
#include "sde.hpp"
#include "schemes.hpp"
#include "multidim.hpp"
#include "optimlib/optim.hpp"
using namespace std;


template <typename TAlgo, typename TRandom, int d>
struct random_walk_md {
    using result_type = typename TAlgo::result_type;

    random_walk_md() = default;

    random_walk_md(TAlgo algo, TRandom Z, unsigned n_max, double barrier, double delta)
        : algo(algo), Z(Z),  algo_init(algo), n(0), n_max(n_max), barrier(barrier), knocked(false),
        sigma_max(algo.getSde().sigma().max()), delta(delta) { } //sigma is constant, and is the same for all libors

    void init() { n = 0; algo = algo_init; knocked = false;}

    bool not_end() { return ((n < n_max) && (!knocked)); }

    double Rswap(vec<d> state){ // state : vector of log libors
      vec<d> denom;
      denom.fill(1 + delta * exp(state[0]));
      for (int i = 1; i < d; i++){
        denom[i] = denom[i-1] * (1 + delta * exp(state[i]));
      }
      double S = 0;
      for (auto i : denom){
        S += 1. / i;
      }
      S *= delta;

      return (1 - 1./denom[d-1])/S;
    }


    bool easyBoundaryCondition(double lHat){
      return (lHat < barrier);
    }

    bool expensiveBoundaryCondition(){
      double sigma_max_k = sigma_max; //sigma is constant
      vec<d> maxState;
      vec<d> state = algo.getState();
      for (int k = 0; k < d; k++){
        maxState[k] = log(exp(state[k]) * (1 + (k+1) * algo.getSde().sigma()[k] * sigma_max_k * algo.getH()
                      + sqrt(algo.getH()) * algo.getSde().sigma()[k]));
      }
      return (Rswap(maxState) < exp(barrier));
    }


    double stateMax() const { return algo.getState().max();}

    vec<d> minimise(){
      vec<d> state = algo.getState();
      arma::vec stateOpt(d-1);
      for (int i = 1; i < d; i++){
        stateOpt[i-1] = state[i];
      }
      arma::vec params(d-1);
      for (int i = 0; i < d-1; i++){
        params[i] = state[i+1];
      }


      std::function<double(const arma::vec &, arma::vec*,  void*)> objective = {[=] (const arma::vec &libor, arma::vec* grad_out, void* opt_data) {
        // Libor = L_1, .., L_(d-1)

        double distance =  dot(stateOpt - libor, stateOpt - libor);

        double prod = 1.0;
        double s = 1.0;
        for(int i=1; i<d; i++){
            prod *= (1+delta* exp(libor[d-1-i]));
            s += prod;
        }
        double L = ((1 + delta * exp(barrier) * s) / (delta * prod)) - 1./delta;
        //cout << "L = " << L << endl;
        distance += (L - state[0]) * (L - state[0]);
        //cout << "dist = " << distance << endl;
        return distance;
        }
      };


      optim::bfgs(params, objective, nullptr);


      double prod = 1.0;
      double s = 0.0;
      for(int i=0;i<d; i++){
          prod *= (1+delta* exp(params[d-1-i]));
          s += prod;
      }
      double L = log((exp(barrier) * (1 + s)+1)/prod - 1.0/delta);

      vec<d> newState;
      for (int i = 1; i < d; i++){
        newState[i] = params[i-1];
      }
      newState[0] = L;

      return newState;
      }

    operator result_type() const { return algo.getState(); }

    result_type st() const { return algo.getState(); }


    random_walk_md & next(vec<d> const & realization) {
      // Simple Monte Carlo
      /*
      ++n;
      algo(realization);
      if (Rswap(algo.getState()) >= exp(barrier)) knocked = true;
      */

      // Algo 1

      vec<d> state = algo.getState();
      double stateMaxValue = stateMax();
      double lHat = stateMaxValue + sigma_max * sigma_max * algo.getH() * d
                          - sigma_max * sigma_max * algo.getH() / 2. + sigma_max * sqrt(algo.getH() * d);

      if (easyBoundaryCondition(lHat)){
        ++n;
        algo(realization);
      }
      else{
        if (expensiveBoundaryCondition()){
          ++n;
          algo(realization);
        }
        else{
          vec<d> projection = minimise();
          random_device rd;
          auto seed = rd();
          mt19937_64 gen(seed);

          double step = sqrt(d) * (sigma_max * sigma_max * algo.getH() * (d - 0.5) + sigma_max * sqrt(algo.getH()*d));
          double proba = step / (step + norm(projection - state, 2)); // Verify norm
          bernoulli_distribution G(proba);
          if (G(gen)){
            knocked = true;
          }
          else{
            algo.setState(state + step * (state - projection) / norm(state - projection, 2)); // verify norm
            ++n;
            algo(realization);
          }
        }
      }
        return *this;
    }

    bool getKnocked() const { return knocked;};
    double getDelta() const { return delta;};
    unsigned getNmax() const { return n_max;};
    TAlgo getAlgo_init() const { return algo_init;};
    TAlgo getAlgo() const { return algo;};
    template <typename TGen>
    random_walk_md<TAlgo, TRandom, d> & operator()(TGen & gen);
public:
    TAlgo algo;
    TRandom Z;
protected:
    TAlgo algo_init;
    unsigned n, n_max;
    double barrier; // log of the barrier
    bool knocked;
    double sigma_max, delta; //sigma_max = max (sigma is supposed constant)
};

template <typename TAlgo, typename TRandom, int d>
template <typename TGen>
random_walk_md<TAlgo, TRandom, d> & //::result_type
random_walk_md<TAlgo, TRandom, d>::operator()(TGen & gen) { // Optimisation fails sometimes
    init();
    while (not_end()) {
      next(Z(gen));
    }
    while (algo.getState()[0] != algo.getState()[0]){ // Libor = NaN when the optimisation fails
      init();
      while (not_end()) {
        next(Z(gen));
      }
    }
    return *this;
};


template <typename TAlgo, typename TRandom, int d>
random_walk_md<TAlgo, TRandom, d>
make_random_walk_md(TAlgo algo, TRandom Z, unsigned n_max, double barrier, double delta) {
    return { algo, Z, n_max, barrier, delta };
};

struct BarrierSwap {
    BarrierSwap() = default;

    BarrierSwap(double K)
        : K(K) {};
    template <typename TAlgo, typename TRandom, int d>
    double operator()(random_walk_md<TAlgo, TRandom, d> & algo) const {
      if (algo.getKnocked()) {cout << "knocked path\n"; return 0;};
      double SwapR = algo.Rswap(algo.getAlgo().getState());
      if (SwapR <= K) {cout << "less than K\n"; return 0;};//;
      double term = 1;
      double sum = 0;
      for (auto i : algo.getAlgo().getState()){
        term = term / (1 + algo.getDelta() * exp(i));
        sum += term;
      }
      double initPrice = 1. / pow(1 + exp(algo.getAlgo_init().getState()[0]) * algo.getDelta(), algo.getNmax() * algo.getAlgo().getH());
      double payoff = initPrice * algo.getDelta() * (SwapR - K) * sum;
      cout << "Path payoff = " << payoff << endl;
      return payoff;
    };

    private:
        double K;
};
