#pragma once
#include <utility>

template <typename Outer, typename Inner>
struct composed
{
    composed() = default;
    composed(Outer f, Inner g) : f(f), g(g) { }

    template<typename T>
    auto operator()(T && x) -> decltype(Outer()(Inner()(x))) {
        return f(g(x));
    }

    private:
        Outer f;
        Inner g;
};

template <typename Outer, typename Inner>
inline composed<Outer, Inner> compose(Outer f, Inner g){
    return composed<Outer, Inner>(f, g);
}

template <class F, typename... Fs>
inline composed<F, composed<Fs...>> compose(F  f, Fs... fs) {
  return compose(f, compose(fs...));
}
