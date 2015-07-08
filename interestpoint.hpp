#ifndef INTERESTPOINT_HPP
#define INTERESTPOINT_HPP

#include <set>

#include "types.hpp"
#include "point.hpp"

namespace sift {
    struct InterestPoint {
        public:
            f32_t scale;
            u16_t octave;
            bool filtered = false;
            Point<u16_t, u16_t> loc;
            std::set<f32_t> orientation;

            InterestPoint() = default;
            explicit InterestPoint(Point<u16_t, u16_t> loc, f32_t scale, u16_t octave) :  scale(scale), octave(octave), loc(loc){
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
