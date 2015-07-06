#ifndef SIFT_HPP
#define SIFT_HPP

#include <cmath>
#include <array>
#include <vector>

#include <vigra/multi_array.hxx>
#include <vigra/matrix.hxx>

#include "types.hpp"
#include "matrix.hpp"
#include "octaveelem.hpp"
#include "interestpoint.hpp"

namespace sift {
    class Sift {
        private:
            const f32_t _sigma;
            const f32_t _k;
            const u16_t _dogsPerEpoch;
            const u16_t _octaves;
            Matrix<OctaveElem> _gaussians;

        public:
            /*
             * @param with standard value 1.6
             * @param with standard value square root of 2
             * @param How many DOGs should be created per octave
             * @param with how many octaves should be calculated
             */
            explicit 
                Sift(u16_t dogsPerEpoch = 3, u16_t octaves = 3, f32_t sigma = 1.6, f32_t k = std::sqrt(2)) : 
                    _sigma(sigma), _k(k), _dogsPerEpoch(dogsPerEpoch), _octaves(octaves) {
                    }

            /*
             * Processes the whole Sift calculation
             * @param the given image
             */
            void calculate(vigra::MultiArray<2, f32_t>&);

        private:
            /*
             * Keypoint Location uses Taylor expansion to filter the weak interest points.
             * @param the vector with octaves and the keypoints as tuples inside the octaves, which will be
             * filtered
             */
            void _eliminateEdgeResponses(std::vector<InterestPoint>&,  const Matrix<OctaveElem>&) const;

            /*
             * Searches for the highest Element in the orientation histogram and searches for other 
             * orientations within a 80% range. Everything outside the range will be set to -1.
             * @param histo The given histogram on which the peak calculation finds place
             */
            const std::array<f32_t, 36> _findPeaks(const std::array<f32_t, 36>&) const;

            /**
             * Calculates the orientation assignments for the interestPoints
             * @param interestPoints the found interestPoints
             * @param dogs the Difference of Gaussians
             */     
            void _orientationAssignment(std::vector<InterestPoint>&);

            /**
             * Finds the nearest gaussian, based on the scale given
             * @param scale the scale
             * @return the nearest gaussian
             */
            const OctaveElem& _findNearestGaussian(f32_t);

            /*
             * Finds the Scale space extrema.
             * @param a vector of vectors of DOGs
             */
            const std::vector<InterestPoint> _findScaleSpaceExtrema(const Matrix<OctaveElem>& dogs) const;

            /*
             * Creates the Laplacians of Gaussians for the count of octave.
             * @param the given img
             * @return a vector with the octtave, which contains DOGs
             */
            const Matrix<OctaveElem> _createDOGs(vigra::MultiArray<2, f32_t>&);
    };
}
#endif //SIFT_HPP
