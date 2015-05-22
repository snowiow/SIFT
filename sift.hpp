#ifndef SIFT_HPP
#define SIFT_HPP

#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>

#include <vigra/convolution.hxx>
#include <vigra/resizeimage.hxx>
#include <vigra/impex.hxx>
#include <vigra/multi_array.hxx>
#include <vigra/multi_math.hxx>

class Sift {

public:
    /*
    * Processes the whole Sift calculation
    * @param img the given image
    * @param sigma with standard value 1.6
    * @param k with standard value square root of 2
    */
    void calculate(vigra::MultiArray<2, float> img, float sigma = 1.6,
        float k= std::sqrt(2));

private:
    /*
    * Resamples an image by 0.5
    * @param in the input image
    * @return the output image
    */
    vigra::MultiArray<2, float> reduceToNextLevel(
        const vigra::MultiArray<2, float>& in);

    /**
    * Convolves a given image with gaussian with a given sigma
    * @param input the input image which will be convolved
    * @sigma the standard deviation for the gaussian
    * @return blured image
    */
    vigra::MultiArray<2, float> convolveWithGauss(
        const vigra::MultiArray<2, float>& input, float sigma);

    /**
    * Calculates the Laplacian of Gaussian, which is the differnce between 2
    * images which were convolved with gaussian under usage of a constant K
    * @param lower the image which lies lower in an octave
    * @param higher the image which lies higher in an octave
    * @return the laplacian of gaussian image, which contains our interest points
    */
    vigra::MultiArray<2, float> laplacianOfGaussian(
        const vigra::MultiArray<2, float>& lower,
        const vigra::MultiArray<2, float> higher);
};

#endif //SIFT_HPP
