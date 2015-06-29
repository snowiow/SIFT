#ifndef SIFT_HPP
#define SIFT_HPP

#include <cmath>
#include <vigra/multi_array.hxx>
#include <memory>

#include "types.hpp"
#include "matrix.hpp"

class Sift {
private:
    f32_t _sigma;
    f32_t _k;
    u16_t _dogsPerEpoch;
    u16_t _epochs;
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

    /**
     * Calculates the orientation assignments for the interestPoints
     * @param interestPoints the found interestPoints
     */     
    void _orientationAssignment(Matrix<Matrix<f32_t>>);

    /**
     * Calculates the scale of the given image data 
     * @param epoch the epoch of the img
     * @param index the index in the current epoch
     * @param sigma the standard derivation of the gaussian
     * @param k the constant which is calculated onto sigma
     * @param dogsPerEpoch the overall count of elems per epoch
     * @return the scale for the given data
     */
    f32_t _calculateScale(u16_t epoch, u16_t index, f32_t sigma, f32_t k, u16_t dogsPerEpoch) const;

    /*
    * Finds the Scale space extrema.
    * @param a vector of vectors of DOGs
    */
    const Matrix<Matrix<f32_t>> _findScaleSpaceExtrema(const Matrix<vigra::MultiArray<2, f32_t>>& dogs) const;

    /*
     * Calculates the first order derivative of the image, at the coordinates
     * @param img the image of which the first derivative is taken.
     * @param p the point at which the derivative is taken
     * @return the derivative as a vector (dx, dy, ds) 
     */
    const vigra::TinyVector<f32_t, 3> _foDerivative(const vigra::MultiArray<2, f32_t>[3], const Point&) const;

    /*
     * Calculates the second order derivative of the image, at the coordinates
     * @param img the image of which the second derivative is taken.
     * @param p the point at which the derivative is taken
     * @return the derivative as a matrix 
     * (dxx, dxy, dxs)
     * (dyx, dyy, dys)
     * (dsx, dsy, dss) 
     */
    const vigra::MultiArray<2, f32_t> _soDerivative(const vigra::MultiArray<2, f32_t>[3], const Point&) const;

    /*
    * Creates the Laplacians of Gaussians for the count of epochs.
    * @param the given img
    * @return a vector with the epochs, which contains DOGs
    */
    const Matrix<vigra::MultiArray<2, f32_t>> _createDOGs(vigra::MultiArray<2, f32_t>&);

    /*
    * Resamples an image by 0.5
    * @param in the input image
    * @return the output image
    */
    const vigra::MultiArray<2, f32_t> _reduceToNextLevel(const vigra::MultiArray<2, f32_t>&, 
            f32_t) const;

    /**
    * Convolves a given image with gaussian with a given sigma
    * @param input the input image which will be convolved
    * @sigma the standard deviation for the gaussian
    * @return blured image
    */
    const vigra::MultiArray<2, f32_t> _convolveWithGauss(const vigra::MultiArray<2, f32_t>&, 
            f32_t) const;

    /**
    * Calculates the Laplacian of Gaussian, which is the differnce between 2
    * images which were convolved with gaussian under usage of a constant K
    * @param lower the image which lies lower in an octave
    * @param higher the image which lies higher in an octave
    * @return the laplacian of gaussian image, which contains our interest points
    */
    const vigra::MultiArray<2, f32_t> _dog(const vigra::MultiArray<2, f32_t>&, 
            const vigra::MultiArray<2, f32_t>&) const;
};

#endif //SIFT_HPP
