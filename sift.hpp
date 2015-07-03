#ifndef SIFT_HPP
#define SIFT_HPP

#include <cmath>
#include <array>

#include <vigra/multi_array.hxx>
#include <vigra/matrix.hxx>

#include "types.hpp"
#include "matrix.hpp"

class Sift {
private:
    const f32_t _sigma;
    const f32_t _k;
    const u16_t _dogsPerEpoch;
    const u16_t _epochs;
    FMatrix<vigra::MultiArray<2, f32_t>> _gaussians;

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
    void _eliminateEdgeResponses(FMatrix<FMatrix<f32_t>>&, const FMatrix<vigra::MultiArray<2, f32_t>>&) const;

    /*
     * Creates the orientation Histogram of a given
     * @param orientations The img of which the histogram is taken from. Needs to be computed by gradient 
     * orientations before
     * @param magnitudes The img of which the bins of the histogram will be weighted. Need to be
     * precomputed by gradient magnitude
     * @param scale The scale of the current images
     * @return histogram with 36 bins which are weighted by magnitudes and gaussian
     */
    const std::array<f32_t, 36> _orientationHistogram(const vigra::MultiArray<2, f32_t>&, 
            const vigra::MultiArray<2, f32_t>& , f32_t) const;


    /*
     * Searches for the highest Element in the orientation histogram and searches for other 
     * orientations within a 80% range. Everything outside the range will be set to -1.
     * @param histo The given histogram on which the peak calculation finds place
     */
    void _createPeak(std::array<f32_t, 36>&);

    /**
     * Calculates the gradient magnitude of the given image at the given position
     * @param img the given img
     * @param p the current point
     * @return the gradient magnitude value
     */
    f32_t _gradientMagnitude(const vigra::MultiArray<2, f32_t>&, const Point&) const;

    /**
     * Calculates the gradient orientation of the given image at the given position
     * @param img the given img
     * @param p the current point
     * @return the gradient orientation value
     */
    f32_t _gradientOrientation(const vigra::MultiArray<2, f32_t>&, const Point&) const;

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
    void _orientationAssignment(FMatrix<FMatrix<f32_t>>);

    /*
     * Finds the Scale space extrema.
     * @param a vector of vectors of DOGs
     */
    const FMatrix<FMatrix<f32_t>> _findScaleSpaceExtrema(const FMatrix<vigra::MultiArray<2, f32_t>>& dogs) const;

    /*
     * Calculates the first order derivative of the image, at the coordinates
     * @param img the image of which the first derivative is taken.
     * @param p the point at which the derivative is taken
     * @return the derivative as a vector (dx, dy, ds) 
     */
    const vigra::Matrix<f32_t> _foDerivative(const vigra::MultiArray<2, f32_t>[3], const Point&) const;

    /*
     * Calculates the second order derivative of the image, at the coordinates
     * @param img the image of which the second derivative is taken.
     * @param p the point at which the derivative is taken
     * @return the derivative as a matrix 
     * (dxx, dxy, dxs)
     * (dyx, dyy, dys)
     * (dsx, dsy, dss) 
     */
    const vigra::Matrix<f32_t> _soDerivative(const vigra::MultiArray<2, f32_t>[3], const Point&) const;

    /*
     * Creates the Laplacians of Gaussians for the count of epochs.
     * @param the given img
     * @return a vector with the epochs, which contains DOGs
     */
    const FMatrix<vigra::MultiArray<2, f32_t>> _createDOGs(vigra::MultiArray<2, f32_t>&);

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
