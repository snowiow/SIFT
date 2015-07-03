#ifndef POINT_HPP
#define POINT_HPP

#include "types.hpp"

namespace sift {
    struct Point {
        u32_t x;
        u32_t y;

        Point(u32_t x, u32_t y) : x(x), y(y) {

        }
    };
}
#endif //POINT_HPP
