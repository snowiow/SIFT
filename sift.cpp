#include "sift.hpp"

#include <iostream>
#include <vector>
#include <string>

#include <vigra/convolution.hxx>
#include <vigra/resizeimage.hxx>
#include <vigra/impex.hxx>
#include <vigra/multi_array.hxx>
#include <vigra/multi_math.hxx>

using namespace vigra::multi_math;

void Sift::calculate(vigra::MultiArray<2, f32_t>& img, f32_t sigma, f32_t k) {
    std::vector<vigra::MultiArray<2, f32_t>> laplacians;
    std::vector<vigra::MultiArray<2, f32_t>> gaussians;
    gaussians.emplace_back(convolveWithGauss(img, sigma));

    //first epoch
    const u32_t laplacionsPerEpoch = 4;

    for (u32_t i = 1; i < laplacionsPerEpoch + 1; i++) {
        gaussians.emplace_back(convolveWithGauss(img, std::pow(k, i) * sigma));
        laplacians.emplace_back(laplacianOfGaussian(gaussians[i - 1], gaussians[i]));
    }

    //Save laplacianOfGaussian for viewing
    for (u32_t i = 0; i < laplacians.size(); i++) {
        std::string fnStr = "laplacian" + std::to_string(i) + ".png";
        exportImage(laplacians[i], vigra::ImageExportInfo(fnStr.c_str()));
    }
}

vigra::MultiArray<2, f32_t> Sift::reduceToNextLevel(const vigra::MultiArray<2, f32_t> & in) {
    // image size at current level
    const i32_t height = in.height();
    const i32_t width = in.width();

    // image size at next smaller level
    const i32_t newheight = (height + 1) / 2;
    const i32_t newwidth = (width + 1) / 2;

    // resize result image to appropriate size
    vigra::MultiArray<2, f32_t> out(vigra::Shape2(newwidth, newheight));

    // define a Gaussian kernel (size 5x1)
    vigra::Kernel1D<f64_t> filter;
    //Inits the filter with the 5 values from -2 the value to +2 the value
    filter.initExplicitly(-2, 2) = 0.05, 0.25, 0.4, 0.25, 0.05;

    vigra::MultiArray<2, f32_t> tmpimage1(vigra::Shape2(width, height));
    vigra::MultiArray<2, f32_t> tmpimage2(vigra::Shape2(width, height));

    //convolves every row from in to out with the filter
    separableConvolveX(in, tmpimage1, filter);
    //same for y
    separableConvolveY(tmpimage1, tmpimage2, filter);
    // downsample smoothed image
    resizeImageNoInterpolation(tmpimage2, out);

    return out;
}

vigra::MultiArray<2, f32_t> Sift::convolveWithGauss(const vigra::MultiArray<2, f32_t>& input, f32_t sigma) {
    vigra::Kernel1D<f64_t> filter;
    filter.initGaussian(sigma);

    vigra::MultiArray<2, f32_t> tmp(input.shape());
    vigra::MultiArray<2, f32_t> result(input.shape());

    separableConvolveX(input, tmp, filter);
    separableConvolveY(tmp, result, filter);

    return result;
}

vigra::MultiArray<2, f32_t> Sift::laplacianOfGaussian(const vigra::MultiArray<2, f32_t>& lower, const vigra::MultiArray<2, f32_t>& higher) {
    return higher - lower;
}
