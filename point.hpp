#ifndef POINT_HPP
#define POINT_HPP

#include "types.hpp"

namespace sift {
    template <typename T, typename U>
        /**
         * A simple Point class which can have two different types for the two coordinates.
         */
        class Point {
            public:
            /**
             * The horizontal coordinate
             */
            T x;

            /**
             * The vertical coordinate
             */
            U y;

            Point() = default;
            Point(T x, U y) : x(x), y(y) {

            }
        };
}
#endif //POINT_HPP
