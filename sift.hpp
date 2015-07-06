#ifndef SIFT_HPP
#define SIFT_HPP

#include <cmath>
#include <array>

#include <vigra/multi_array.hxx>
#include <vigra/matrix.hxx>

#include "types.hpp"
#include "matrix.hpp"

namespace sift {
    class Sift {
        private:
            const f32_t _sigma;
            const f32_t _k;
            const u16_t _dogsPerEpoch;
            const u16_t _epochs;
            Matrix<vigra::MultiArray<2, f32_t>> _gaussians;

        public:
            /*
             * @param with standard value 1.6
             * @param with standard value square root of 2
             * @param How many DOGs should be created per epoch
             * @param with how many epochs should be calculated
             */
            explicit 
                Sift(f32_t sigma = 1.6, f32_t k = std::sqrt(2), u16_t dogsPerEpoch = 3, u16_t epochs = 3) : 
                    _sigma(sigma), _k(k), _dogsPerEpoch(dogsPerEpoch), _epochs(epochs) {
                    }

            /*
             * Processes the whole Sift calculation
             * @param the given image
             */
            void calculate(vigra::MultiArray<2, f32_t>&);

        private:
            /*
             * Keypoint Location uses Taylor expansion to filter the weak interest points.
             * @param the vector with epochs and the keypoints as tuples inside the epochs, which will be
             * filtered
             */
            void _eliminateEdgeResponses(Matrix<Matrix<f32_t>>&, const Matrix<vigra::MultiArray<2, f32_t>>&) const;

            /*
             * Searches for the highest Element in the orientation histogram and searches for other 
             * orientations within a 80% range. Everything outside the range will be set to -1.
             * @param histo The given histogram on which the peak calculation finds place
             */
            const std::array<f32_t, 36> _findPeaks(const std::array<f32_t, 36>&) const;

            /*
             * Calculates the scale based on the given parameters
             * @param epoch the current epoch of the img
             * @param i the current index in the epoch
             */
            f32_t _calculateScale(u16_t, u16_t) const;

            /**
             * Calculates the orientation assignments for the interestPoints
             * @param interestPoints the found interestPoints
             * @param dogs the Difference of Gaussians
             */     
            void _orientationAssignment(Matrix<Matrix<f32_t>>);

            /*
             * Finds the Scale space extrema.
             * @param a vector of vectors of DOGs
             */
            const Matrix<Matrix<f32_t>> _findScaleSpaceExtrema(const Matrix<vigra::MultiArray<2, f32_t>>& dogs) const;

            /*
             * Creates the Laplacians of Gaussians for the count of epochs.
             * @param the given img
             * @return a vector with the epochs, which contains DOGs
             */
            const Matrix<vigra::MultiArray<2, f32_t>> _createDOGs(vigra::MultiArray<2, f32_t>&);
    };
}
#endif //SIFT_HPP
