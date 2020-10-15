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


using namespace std;
int main() {
    using generator = std::mt19937_64;
    generator gen(12398456);

    double r = 0.1, sigma = 0.2;
    double x0 = 100;

    double T = 1;
    unsigned N = 1e3;
    std::normal_distribution<> G;
    std::cout << "# Euler (exact)" << std::endl;
    for (int n = 4; n < 1025; n *= 2) {
        auto h = T / (double) n;
        auto X_h = euler<BS>({r, sigma, x0}, h);
        auto X_n = make_random_scheme(X_h, G, n);
        auto X_exact = make_random_scheme(exact_bs(r, sigma, x0, h), G, n);
        auto E = make_strong_error(X_n, X_exact);
        auto mc = monte_carlo(E, gen, N);
        std::cout << n << "\t" << h << "\t" << sqrt(mc.mean())
            << "\t" << sqrt(mc.mean() - mc.ic_size())
            << "\t" << sqrt(mc.mean() + mc.ic_size()) << std::endl;
    }
    std::cout << "\n\n";
    std::cout << "# Milstein (exact)" << std::endl;
    for (int n = 4; n < 1025; n *= 2) {
        auto h = T / (double) n;
        auto X_h = milstein<BS>({r, sigma, x0}, h);
        auto X_n = make_random_scheme(X_h, G, n);
        auto X_exact = make_random_scheme(exact_bs(r, sigma, x0, h), G, n);
        auto E = make_strong_error(X_n, X_exact);
        auto mc = monte_carlo(E, gen, N);
        std::cout << n << "\t" << h << "\t" << sqrt(mc.mean())
            << "\t" << sqrt(mc.mean() - mc.ic_size())
            << "\t" << sqrt(mc.mean() + mc.ic_size()) << std::endl;
    }
    std::cout << "\n\n";
    std::cout << "# Euler (approx)" << std::endl;
    for (int n = 4; n < 1025; n *= 2) {
        auto h = T / (double) n;
        auto X_h = euler<BS>({r, sigma, x0}, h);
        auto X_n = make_random_scheme(X_h, G, n);
        auto E = make_strong_error(X_n);
        auto mc = monte_carlo(E, gen, N);
        std::cout << n << "\t" << h << "\t" << sqrt(mc.mean())
            << "\t" << sqrt(mc.mean() - mc.ic_size())
            << "\t" << sqrt(mc.mean() + mc.ic_size()) << std::endl;
    }
    std::cout << "\n\n";
    std::cout << "# Milstein (approx)" << std::endl;
    for (int n = 4; n < 1025; n *= 2) {
        auto h = T / (double) n;
        auto X_h = milstein<BS>({r, sigma, x0}, h);
        auto X_n = make_random_scheme(X_h, G, n);
        auto E = make_strong_error(X_n);
        auto mc = monte_carlo(E, gen, N);
        std::cout << n << "\t" << h << "\t" << sqrt(mc.mean())
            << "\t" << sqrt(mc.mean() - mc.ic_size())
            << "\t" << sqrt(mc.mean() + mc.ic_size()) << std::endl;
    }
    std::cout << "\n\n";

    auto model = CIR(1, 1, 1, 1);
    std::cout << "# Euler (approx)" << std::endl;
    for (int n = 4; n < 1025; n *= 2) {
        auto h = T / (double) n;
        auto X_h = euler<CIR>(model, h);
        auto X_n = make_random_scheme(X_h, G, n);
        auto E = make_strong_error(X_n);
        auto mc = monte_carlo(E, gen, N);
        std::cout << n << "\t" << h << "\t" << sqrt(mc.mean())
            << "\t" << sqrt(mc.mean() - mc.ic_size())
            << "\t" << sqrt(mc.mean() + mc.ic_size()) << std::endl;
    }
    std::cout << "\n\n";
    std::cout << "# Milstein (approx)" << std::endl;
    for (int n = 4; n < 1025; n *= 2) {
        auto h = T / (double) n;
        auto X_h = milstein<CIR>(model, h);
        auto X_n = make_random_scheme(X_h, G, n);
        auto E = make_strong_error(X_n);
        auto mc = monte_carlo(E, gen, N);
        std::cout << n << "\t" << h << "\t" << sqrt(mc.mean())
            << "\t" << sqrt(mc.mean() - mc.ic_size())
            << "\t" << sqrt(mc.mean() + mc.ic_size()) << std::endl;
    }
    std::cout << "\n\n";
    std::cout << "# Implicit" << std::endl;
    for (int n = 4; n < 1025; n *= 2) {
        auto h = T / (double) n;
        auto X_h = implicit_CIR(model, h);
        auto X_n = make_random_scheme(X_h, G, n);
        auto E = make_strong_error(X_n);
        auto mc = monte_carlo(E, gen, N);
        std::cout << n << "\t" << h << "\t" << sqrt(mc.mean())
            << "\t" << sqrt(mc.mean() - mc.ic_size())
            << "\t" << sqrt(mc.mean() + mc.ic_size()) << std::endl;
    }

    return 0;
}
