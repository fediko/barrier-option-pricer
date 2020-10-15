#include <iostream>
#include <vector>
#include <random>
#include <iterator>
#include "compose.hpp"
#include "mc.hpp"
#include "sde.hpp"
#include "schemes.hpp"
#include "schemes_BS.hpp"
#include "schemes_CIR.hpp"
#include "strong_error.hpp"
#include "multidim.hpp"
#include "BarCap.hpp"
#include "BarSwap.hpp"

using namespace std;
int main() {
    random_device rd;
    auto seed = rd();
    mt19937_64 gen(seed);

    string option;
    cout << "option = ? (S for swaption, C for caplet) " << endl;
    cin >> option;
    if (option == "C"){

      // Caplet
      double r = 0., sigma = 0.25;
      double x0 = log(0.13);
      double T = 9;
      unsigned N = 1e3; //Sample size
      unsigned n = 100; //Number of steps

      bernoulli_distribution U(0.5);
      auto h = T / (double) n;
      function<double(double)> b = {[=] (double x) { return r -  sigma * sigma / 2;}};
      function<double(double)> sig = {[=] (double x) { return sigma;}};
      auto X_h = euler<sde<>>({b, sig, x0}, h);
      auto F = make_random_walk(X_h, U, n, log(0.28));
      BarrierCap payoff(0.01);
      auto Y = compose(payoff, F);
      auto mc = monte_carlo(Y, gen, N);
      std::cout << N << "\t" << h << "\t" << mc.mean()
          << "\t" << mc.mean() - mc.ic_size()
          << "\t" << mc.mean() + mc.ic_size() << std::endl;

      }
    // Swaption
    else {
      double sigma = 0.1;
      double T = 10;
      unsigned Tend = 20;
      double barrier = 0.075;
      double strike = 0.01;
      double betha = 0.1;
      const int d = 10;
      double delta = (Tend - T) / (double) d;
      vec<d> x0;
      x0.fill(log(0.05));
      unsigned N = 1e3; //Sample size
      unsigned n = 100; //Number of steps
      auto h = T / (double) n;

      mat<d> correl;
      for (int i = 0; i < d; i++){
        for (int j = 0; j < d; j++){
          correl.at(i, j) = exp(-betha * abs(i-j) * delta);
        }
      }
      mat<d> G(chol(correl, "lower"));

      //G.print();

      bernoulli_dist_md<d> U(G, 0.5);
      function<vec<d>()> sig = {[sigma] () { vec<d> sigmaVec; sigmaVec.fill(sigma); return sigmaVec;}};
      function<vec<d>(vec<d>)> b = {[correl, sig, delta] (vec<d> state) {
        vec<d> drift;

        for (int i = 0; i < d; i++){
          double sigma_i = sig()[i];
          double s = 0;
          double num = 1;
          double denom = 1;
          for (int j = 0; j <= i; j++){
            num = delta * exp(state[j]);
            denom = (1 + delta * exp(state[j]));
            s += (num / denom) * correl.at(i, j) * sig()[j];
          }
          drift[i] = sigma_i * s - sigma_i * sigma_i / 2.;

        }
        return drift;
      }

      };
      euler_md<d> X_h = euler_md<d>({x0, b, sig}, h);
      //random_walk_md<euler_md<d>, bernoulli_dist_md<d>, d> F = make_random_walk_md(X_h, U, n, log(barrier), delta);
      random_walk_md<euler_md<d>, bernoulli_dist_md<d>, d> F = random_walk_md<euler_md<d>, bernoulli_dist_md<d>, d> (X_h, U, n, log(barrier), delta);
      BarrierSwap payoff(strike);
      auto Y = compose(payoff, F);
      auto mc = monte_carlo(Y, gen, N);
      std::cout << N << "\t" << h << "\t" << mc.mean()
          << "\t" << mc.mean() - mc.ic_size()
          << "\t" << mc.mean() + mc.ic_size() << std::endl;
      }

      return 0;
}
