#include "sift.hpp"

#include <iostream>
#include <vector>
#include <string>

#include <vigra/convolution.hxx>
#include <vigra/resizeimage.hxx>
#include <vigra/impex.hxx>
#include <vigra/multi_math.hxx>

using namespace vigra::multi_math;

void Sift::calculate(vigra::MultiArray<2, f32_t>& img, u16_t epochs,
    f32_t sigma, f32_t k) {

    //How many laplacianOfGaussians per epoch
    const u32_t laplacionsPerEpoch = 4;

    /*
    * A vector which contains a vector for each epoch. Those are filled with the
    * Images which were created from the laplacianOfGaussians.
    */
    std::vector<std::vector<vigra::MultiArray<2, f32_t>>> laplacians;

    /*
    * a vector which contains a vector for each epoch. Those are filled with the
    * images which were created from gaussians
    */
    std::vector<std::vector<vigra::MultiArray<2, f32_t>>> gaussians;

    //add inital image
    gaussians.emplace_back(std::vector<vigra::MultiArray<2, f32_t>>());
    gaussians[0].emplace_back(convolveWithGauss(img, sigma));

    //TODO: More elegant way?
    u32_t exp = 0;
    for (u32_t i = 0; i < epochs; i++) {
        laplacians.emplace_back(std::vector<vigra::MultiArray<2, f32_t>>());
        for (u32_t j = 1; j < laplacionsPerEpoch + 1; j++) {
            std::cout << "exp: " << exp << std::endl;
            gaussians[i].emplace_back(convolveWithGauss(gaussians[i][j - 1],
                std::pow(k, exp) * sigma));
            laplacians[i].emplace_back(laplacianOfGaussian(
                gaussians[i][j - 1], gaussians[i][j]));

            exp++;
        }
        /*
        * If we aren't in the last epoch populate the next level with the second
        * last element, scaled won by a half, of the gaussians.
        */
        if (i < epochs - 1) {
            gaussians.emplace_back(std::vector<vigra::MultiArray<2, f32_t>>());
            auto scaledElem = reduceToNextLevel(
                gaussians[i][laplacionsPerEpoch - 1], sigma);
            gaussians[i + 1].emplace_back(scaledElem);

            exp -= 2;
        }
    }

    //Save laplacianOfGaussian for viewing
    for (u32_t i = 0; i < laplacians.size(); i++) {
        for (u32_t j = 0; j < laplacians[i].size(); j++) {
        std::string fnStr = "images/laplacian" + std::to_string(i) + std::to_string(j)
            + ".png";
        exportImage(laplacians[i][j], vigra::ImageExportInfo(fnStr.c_str()));
        }
    }
}

vigra::MultiArray<2, f32_t> Sift::reduceToNextLevel(const vigra::MultiArray<2, f32_t>& in, f32_t sigma) {
    // image size at current level
    const i32_t height = in.height();
    const i32_t width = in.width();

    // image size at next smaller level
    const i32_t newheight = (height + 1) / 2;
    const i32_t newwidth = (width + 1) / 2;

    // resize result image to appropriate size
    vigra::MultiArray<2, f32_t> out(vigra::Shape2(newwidth, newheight));

    // downsample smoothed image
    resizeImageNoInterpolation(convolveWithGauss(in, sigma), out);
    return out;
}

vigra::MultiArray<2, f32_t> Sift::convolveWithGauss(
    const vigra::MultiArray<2, f32_t>& input, f32_t sigma) {

    vigra::Kernel1D<f64_t> filter;
    filter.initGaussian(sigma);

    vigra::MultiArray<2, f32_t> tmp(input.shape());
    vigra::MultiArray<2, f32_t> result(input.shape());

    separableConvolveX(input, tmp, filter);
    separableConvolveY(tmp, result, filter);

    return result;
}

vigra::MultiArray<2, f32_t> Sift::laplacianOfGaussian(
    const vigra::MultiArray<2, f32_t>& lower,
    const vigra::MultiArray<2, f32_t>& higher) {

    return higher - lower;
}
