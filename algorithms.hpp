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
         * @sigma the standard deviation for the gaussian
         * @return blured image
         */
        const vigra::MultiArray<2, f32_t> convolveWithGauss(const vigra::MultiArray<2, f32_t>&, 
                f32_t);

        /*
         * Resamples an image by 0.5
         * @param in the input image
         * @return the output image
         */
        const vigra::MultiArray<2, f32_t> reduceToNextLevel(const vigra::MultiArray<2, f32_t>&, 
                f32_t);

        /**
         * Calculates the Laplacian of Gaussian, which is the differnce between 2
         * images which were convolved with gaussian under usage of a constant K
         * @param lower the image which lies lower in an octave
         * @param higher the image which lies higher in an octave
         * @return the laplacian of gaussian image, which contains our interest points
         */
        const vigra::MultiArray<2, f32_t> dog(const vigra::MultiArray<2, f32_t>&, 
                const vigra::MultiArray<2, f32_t>&);

        /*
         * Calculates the first order derivative of the image, at the coordinates
         * @param img the image of which the first derivative is taken.
         * @param p the point at which the derivative is taken
         * @return the derivative as a vector (dx, dy, ds) 
         */
        const vigra::Matrix<f32_t> foDerivative(const vigra::MultiArray<2, f32_t>[3], const Point&);

        /*
         * Calculates the second order derivative of the image, at the coordinates
         * @param img the image of which the second derivative is taken.
         * @param p the point at which the derivative is taken
         * @return the derivative as a matrix 
         * (dxx, dxy, dxs)
         * (dyx, dyy, dys)
         * (dsx, dsy, dss) 
         */
        const vigra::Matrix<f32_t> soDerivative(const vigra::MultiArray<2, f32_t>[3], const Point&);

        /**
         * Calculates the gradient magnitude of the given image at the given position
         * @param img the given img
         * @param p the current point
         * @return the gradient magnitude value
         */
        f32_t gradientMagnitude(const vigra::MultiArray<2, f32_t>&, const Point&);

        /**
         * Calculates the gradient orientation of the given image at the given position
         * @param img the given img
         * @param p the current point
         * @return the gradient orientation value
         */
        f32_t gradientOrientation(const vigra::MultiArray<2, f32_t>&, const Point&);

        /*
         * Creates the orientation Histogram of a given
         * @param orientations The img of which the histogram is taken from. Needs to be computed by gradient 
         * orientations before
         * @param magnitudes The img of which the bins of the histogram will be weighted. Need to be
         * precomputed by gradient magnitude
         * @param scale The scale of the current images
         * @return histogram with 36 bins which are weighted by magnitudes and gaussian
         */
        const std::array<f32_t, 36> orientationHistogram(const vigra::MultiArray<2, f32_t>&, 
                const vigra::MultiArray<2, f32_t>& , f32_t);

    }
}
#endif //ALGORITHMS_HPP
