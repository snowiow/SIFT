#ifndef POINT_HPP
#define POINT_HPP

#include "types.hpp"

namespace sift {
    template <typename T, typename U>
    struct Point {
        T x;
        U y;

        Point() = default;
        Point(T x, U y) : x(x), y(y) {

        }
    };
}
#endif //POINT_HPP
