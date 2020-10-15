#pragma once

template <typename TRealScheme1, typename TRealScheme2>
struct strong_error {
    strong_error(TRealScheme1 S1, TRealScheme2 S2)
        : S1(S1), S2(S2), current_max(0) { }

    template <typename TGen>
    double operator()(TGen & gen) {
        S1.init();
        S2.init();
        current_max = 0;
        while (S1.not_end() && S2.not_end()) {
            auto z = S1.Z(gen);
            auto x = S1.next(z) - S2.next(z);
            current_max = std::max(current_max, x * x);
        }
        return current_max;
    }
private:
    TRealScheme1 S1;
    TRealScheme2 S2;
    double current_max;
};

template <typename TRealScheme1, typename TRealScheme2>
strong_error<TRealScheme1, TRealScheme2>
make_strong_error(TRealScheme1 S1, TRealScheme2 S2) {
    return { S1, S2 };
}

template <typename TRealScheme>
struct strong_error_approx {
    strong_error_approx(TRealScheme S)
        : S1(S), S2(S.refine()), current_max(0) { }

    template <typename TGen>
    double operator()(TGen & gen) {
        S1.init();
        S2.init();
        current_max = 0;
        while (S1.not_end() && S2.not_end()) {
            auto z1 = S2.Z(gen);
            auto z2 = S2.Z(gen);
            auto x1 = S1.next((z1 + z2) / sqrt(2.)).st();
            auto x2 = S2.next(z1).next(z2).st();
            auto x = x1 - x2;
            current_max = std::max(current_max, x*x);
        }
        return current_max;
    }
private:
    TRealScheme S1, S2;
    double current_max;
};

template <typename TRealScheme>
strong_error_approx<TRealScheme>
make_strong_error(TRealScheme S) { return { S }; }
