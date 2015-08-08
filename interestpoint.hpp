#ifndef INTERESTPOINT_HPP
#define INTERESTPOINT_HPP

#include <set>

#include "types.hpp"
#include "point.hpp"

namespace sift {
    struct InterestPoint {
        public:
            /**
             * @var the scale of the interestpoint
             */
            f32_t scale;

            /**
             * @var The octave of the interestpoint, because scale alone doesn't identify clearly 
             */
            u16_t octave;
            
            /**
             * @var This is needed because of the fact, the corresponding DoG must not be searched
             * again. And we get a linear access to the corresponding DoG element.
             */
            u16_t index;

            /**
             * @var A flag, which shows if the interestpoint was filtered out by sift
             */
            bool filtered = false;

            /**
             * @var the x and y coordinates of the interest point
             */
            Point<u16_t, u16_t> loc;

            /**
             * @var The orientations of the interest point
             */
            std::set<f32_t> orientation;

            InterestPoint() = default;
            explicit InterestPoint(Point<u16_t, u16_t> loc, f32_t scale, u16_t octave, u16_t index) 
                :  scale(scale), octave(octave), index(index), loc(loc) {
            }

            /**
             * Orders interest points by the fact if they are filtered or not. So the filtered can
             * be deleted from the end of a vector
             */
            static bool cmpByFilter(const InterestPoint &a, const InterestPoint &b) {
                if (!a.filtered && b.filtered) {
                    return true;
                }
                return false;
            }

    };

}
#endif //INTERESTPOINT_HPP
