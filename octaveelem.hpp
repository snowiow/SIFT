#ifndef OCTAVEELEM_HPP
#define OCTAVEELEM_HPP

#include "vigra/multi_array.hxx"
#include "types.hpp"

namespace sift {
    struct OctaveElem {
        public:
            f32_t scale;
            vigra::MultiArray<2, f32_t> img;

            OctaveElem() = default;
    };
}
#endif //OCTAVEELEM_HPP
