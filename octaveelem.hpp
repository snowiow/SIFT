#ifndef OCTAVEELEM_HPP
#define OCTAVEELEM_HPP

#include "vigra/multi_array.hxx"
#include "types.hpp"

namespace sift {
    /**
     * A Class which saves the needed data of the elements in an octave. These are the Gaussians and
     * DoGs.
     */
    class OctaveElem {
        public:
            /**
             * The scale of the current element.
             */
            f32_t scale;

            /**
             * The actual DoG or Gaussian image data.
             */
            vigra::MultiArray<2, f32_t> img;

            OctaveElem() = default;
    };
}
#endif //OCTAVEELEM_HPP
