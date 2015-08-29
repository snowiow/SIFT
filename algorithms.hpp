#ifndef ALGORITHMS_HPP
#define ALGORITHMS_HPP

#include <vigra/multi_array.hxx>
#include <vigra/matrix.hxx>

#include "point.hpp"
#include "types.hpp"

namespace sift {
    namespace alg {
        /**
         * Convolves a given image with gaussian with a given sigma
         * @param input the input image which will be convolved
         * @param sigma the standard deviation for the gaussian
         * @return blured image
         */
        const vigra::MultiArray<2, f32_t> convolveWithGauss(const vigra::MultiArray<2, f32_t>&, 
                f32_t);

        /**
         * Resamples an image by 0.5
         * @param img the input image
         * @return the output image
         */
        const vigra::MultiArray<2, f32_t> reduceToNextLevel(const vigra::MultiArray<2, f32_t>&, 
                f32_t);

        /**
         * Resamples an image by 2
         * @param in the input image
         * @return the output image
         */
        const vigra::MultiArray<2, f32_t> increaseToNextLevel(const vigra::MultiArray<2, f32_t>&,
                f32_t);

        /**
         * Calculates the Difference of Gaussian, which is the differnce between 2
         * images which were convolved with gaussian under usage of a constant K
         * @param lower the image which lies lower in an octave
         * @param higher the image which lies higher in an octave
         * @return the difference of gaussian image, which contains our interest points
         */
        const vigra::MultiArray<2, f32_t> dog(const vigra::MultiArray<2, f32_t>&, 
                const vigra::MultiArray<2, f32_t>&);

        /**
         * Calculates the first order derivative of the image, at the coordinates
         * @param img the image of which the first derivative is taken.
         * @param p the point at which the derivative is taken
         * @return the derivative as a vector (dx, dy, ds) 
         */
        const vigra::Matrix<f32_t> foDerivative(const std::array<vigra::MultiArray<2, f32_t>, 3>&, const Point<u16_t, u16_t>&);

        /**
         * Calculates the second order derivative of the image, at the coordinates
         * @param img the image of which the second derivative is taken.
         * @param p the point at which the derivative is taken
         * @return the derivative as a matrix 
         * (dxx, dxy, dxs)
         * (dyx, dyy, dys)
         * (dsx, dsy, dss) 
         */
        const vigra::Matrix<f32_t> soDerivative(const std::array<vigra::MultiArray<2, f32_t>, 3>&, const Point<u16_t, u16_t>&);

        /**
         * Calculates the gradient magnitude of the given image at the given position
         * @param img the given img
         * @param p the current point
         * @return the gradient magnitude value
         */
        f32_t gradientMagnitude(const vigra::MultiArray<2, f32_t>&, const Point<u16_t, u16_t>&);

        /**
         * Calculates the gradient orientation of the given image at the given position
         * @param img the given img
         * @param p the current point
         * @return the gradient orientation value
         */
        f32_t gradientOrientation(const vigra::MultiArray<2, f32_t>&, const Point<u16_t, u16_t>&);

        /**
         * Creates an orientation Histogram of a given img and his corresponding orientations and 
         * magnitudes. Places values in bins of size 10. So the resulting histogram has 36 elements.
         * @param orientations The img of which the histogram is taken from. Needs to be computed by gradient 
         * orientations before
         * @param magnitudes The img of which the bins of the histogram will be weighted. Need to be
         * precomputed by gradient magnitude
         * @param img the given img
         * @return histogram with 36 bins which are weighted by magnitudes and gaussian
         */
        const std::array<f32_t, 36> orientationHistogram36(const vigra::MultiArray<2, f32_t>&, 
                const vigra::MultiArray<2, f32_t>& , const vigra::MultiArray<2, f32_t>&);

        /**
         * Creates an orientation Histogram of a given img and his corresponding orientations and 
         * magnitudes. Places values in bins of size 45. So the resulting histogram has 8 elements.
         * @param orientations The img of which the histogram is taken from. Needs to be computed by gradient 
         * orientations before
         * @param magnitudes The img of which the bins of the histogram will be weighted. Need to be
         * precomputed by gradient magnitude
         * @param img the given img
         * @return histogram with 8 bins which are weighted by magnitudes and gaussian
         */
        const std::vector<f32_t> orientationHistogram8(const vigra::MultiArray<2, f32_t>&,
                const vigra::MultiArray<2, f32_t>&, const vigra::MultiArray<2, f32_t>&);

        /**
         * Calculates the vertex of a parabola, by taking a max value and its 2 neigbours
         * @param ln the left neighbor of the peak
         * @param peak the peak value
         * @param rn the right neighbor of the peak
         * @return the vertex value
         */
        f32_t vertexParabola(const Point<u16_t, f32_t>&, const Point<u16_t, f32_t>&, 
                const Point<u16_t, f32_t>&);

        /**
         * Rotates a given shape by a given degree clockwise
         * @param center the center point of the shape
         * @param angle by which angle the shape should be rotated
         * @param width the width of the shape
         * @param height the height of the shape
         * @return array with 2 elements. First element represents upper left corner of the shape and
         * bottom right is represented by the second argument
         */
        std::array<Point<f32_t, f32_t>, 4> rotateShape(const Point<u16_t, u16_t>&, f32_t, const u16_t, const u16_t);

        /**
         * Normalizes a vector
         * @param vec the vector to be normalized
         */
        void normalizeVector(std::vector<f32_t>&);
    }
}
#endif //ALGORITHMS_HPP
