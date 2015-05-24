#ifndef SIFT_HPP
#define SIFT_HPP

#include <cmath>
#include <vigra/multi_array.hxx>
#include "types.hpp"

class Sift {
public:
    /*
    * Processes the whole Sift calculation
    * @param img the given image
    * @param sigma with standard value 1.6
    * @param k with standard value square root of 2
    */
    void calculate(vigra::MultiArray<2, f32_t>&, u16_t epochs = 4,
        f32_t sigma = 1.6, f32_t k = std::sqrt(2));

private:
    /*
    * Resamples an image by 0.5
    * @param in the input image
    * @return the output image
    */
    vigra::MultiArray<2, f32_t> reduceToNextLevel(
        const vigra::MultiArray<2, f32_t>&, f32_t);

    /**
    * Convolves a given image with gaussian with a given sigma
    * @param input the input image which will be convolved
    * @sigma the standard deviation for the gaussian
    * @return blured image
    */
    vigra::MultiArray<2, f32_t> convolveWithGauss(
        const vigra::MultiArray<2, f32_t>&, f32_t);

    /**
    * Calculates the Laplacian of Gaussian, which is the differnce between 2
    * images which were convolved with gaussian under usage of a constant K
    * @param lower the image which lies lower in an octave
    * @param higher the image which lies higher in an octave
    * @return the laplacian of gaussian image, which contains our interest points
    */
    vigra::MultiArray<2, f32_t> laplacianOfGaussian(
        const vigra::MultiArray<2, f32_t>&, const vigra::MultiArray<2, f32_t>&);
};

#endif //SIFT_HPP
