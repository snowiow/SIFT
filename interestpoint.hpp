#ifndef INTERESTPOINT_HPP
#define INTERESTPOINT_HPP

#include "types.hpp"
#include "point.hpp"

namespace sift {
    struct InterestPoint {
        public:
            Point<u16_t, u16_t> loc;
            f32_t scale;
            f32_t orientation;
            bool filtered = false;

            InterestPoint() = default;
            explicit InterestPoint(Point<u16_t, u16_t> loc, f32_t scale) : loc(loc), scale(scale) {
            }

            static bool cmpByFilter(const InterestPoint &a, const InterestPoint &b) {
                if (!a.filtered && b.filtered) {
                    return true;
                }
                return false;
            }

    };

}
#endif //INTERESTPOINT_HPP
