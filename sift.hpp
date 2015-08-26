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
        public:
            /**
             * Wether to process the algorithm based on subpixel basis or not
             */
            const bool subpixel;
        private:
            /**
             * The sigma value is used for the standard derivation of the Gaussian calculations.
             */
            const f32_t _sigma;
            /**
             * The constant which is multiplied with sigma  to get the Gaussians and DoGs.
             */
            const f32_t _k;
            
            /**
             * How many DoGs will be calculated per epoch. Minimum is 3.
             */
            const u16_t _dogsPerEpoch;

            /**
             * How many octaves of DoGs will be calculated.
             */
            const u16_t _octaves;

            /**
             * The Gaussians calculated during the algorithm.
             */
            Matrix<OctaveElem> _gaussians;

            /**
             * The magnitudes of the gaussians
             */
            Matrix<vigra::MultiArray<2, f32_t>> _magnitudes;

            /**
             * The orientations of the gaussians
             */
            Matrix<vigra::MultiArray<2, f32_t>> _orientations;

        public:
            /**
             * @param sigma standard value 1.6
             * @param k standard value square root of 2
             * @param dogsPerEpoch How many DOGs should be created per octave
             * @param octaves how many octaves should be calculated
             * @param subpixel wether the calculation is based on subpixel basis or not
             */
            explicit 
                Sift(u16_t dogsPerEpoch = 3, u16_t octaves = 3, f32_t sigma = 1.6, 
                        f32_t k = std::sqrt(2), bool subpixel = false) : 
                        subpixel(subpixel), _sigma(sigma), _k(k), _dogsPerEpoch(dogsPerEpoch), 
                        _octaves(octaves) {
                    }

            /**
             * Processes the whole Sift calculation
             * @param img the given image
             * @return a vector containing the filtered sift features
             */
            std::vector<InterestPoint> calculate(vigra::MultiArray<2, f32_t>&);

        private:
            /**
             * Creates the local image desciptors.
             * @param interestpoints the vector with interestpoints
             */
            void _createDecriptors(std::vector<InterestPoint>&);

            /**
             * Eliminates all values above the threshold of 0.2 and performs a new 
             * vector normalization as long as there are no more values above 0.2
             * @param vec The given vector
             */
            std::vector<f32_t> _eliminateVectorThreshold(std::vector<f32_t>&) const;
            
            /**
             * Creates magnitude versions of all the gaussian images.
             */
            void _createMagnitudePyramid();

            /**
             * Create orientation versions of all the gaussian images.
             */
            void _createOrientationPyramid();

            /**
             * Keypoint Location using Taylor expansion to filter the weak interest points. Those 
             * interest points, which get filtered get their filtered flag set to true
             * @param interestpoints the vector with interestpoints 
             * @param dogs the dogs which were calculated in an earlier step
             */
            void _eliminateEdgeResponses(std::vector<InterestPoint>&,  const Matrix<OctaveElem>&) const;

            /**
             * Searches for the highest Element in the orientation histogram and searches for other 
             * orientations within a 80% range. Everything outside the range will be set to -1.
             * @param histo The given histogram on which the peak calculation finds place
             * @return an array with 36 elements, standing for the 36 bins of orientation possibilities
             */
            const std::set<f32_t> _findPeaks(const std::array<f32_t, 36>&) const;

            /**
             * Calculates the orientation assignments for the interestPoints
             * @param interestPoints the found interestPoints for whom the orientation should be 
             * calulated
             */     
            void _orientationAssignment(std::vector<InterestPoint>&);

            /**
             * Finds the nearest gaussian, based on the scale given
             * @param scale the scale
             * @return the point where the Gaussian is lying in the pyramid
             */
            const Point<u16_t, u16_t> _findNearestGaussian(f32_t);

            /**
             * Finds the Scale space extrema aka InterestPoints
             * @param dogs a matrix of DOGs
             * @param interestPoints a vector which holds interestPoints. Will be filled with the 
             * found interest points
             */
            void _findScaleSpaceExtrema(const Matrix<OctaveElem>&, std::vector<InterestPoint>&) const;

            /**
             * Creates the Difference of Gaussians for the count of octaves.
             * @param img the given img
             * @return a matrix with the octaves as width and octave elements as height, which contain the DoGs
             */
            const Matrix<OctaveElem> _createDOGs(vigra::MultiArray<2, f32_t>&);
    };
}
#endif //SIFT_HPP
