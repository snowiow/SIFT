#include "sift.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <tuple>

#include <vigra/convolution.hxx>
#include <vigra/resizeimage.hxx>
#include <vigra/impex.hxx>
#include <vigra/multi_math.hxx>

using namespace vigra::multi_math;

void Sift::calculate(vigra::MultiArray<2, f32_t>& img, u16_t epochs, f32_t sigma, f32_t k,
    u16_t dogPerEpoch) const {

    auto dogs = _createDOGs(img, epochs, sigma, k, dogPerEpoch);
    auto interestPoints = _findScaleSpaceExtrema(dogs);

    // Save img with found interest points for demonstration purposes
    for (auto point : interestPoints[0]) {
        img(std::get<0>(point), std::get<1>(point)) = 255;
    }

    exportImage(img, vigra::ImageExportInfo("images/interest_points.png"));
}

const std::vector<std::vector<std::tuple<u32_t, u32_t>>>
    Sift::_findScaleSpaceExtrema(std::vector<std::vector<vigra::MultiArray<2, f32_t>>> dogs) const {

    //a Vector with epochs containing tuples of interest points found per epoch
    std::vector<std::vector<std::tuple<u32_t, u32_t>>> interestPoints;
    for (u32_t e = 0; e < dogs.size(); e++) {
        interestPoints.emplace_back(std::vector<std::tuple<u32_t, u32_t>>());
        for (u32_t i = 1; i < dogs[e].size() - 1; i++) {
            for (u32_t x = 0; x < dogs[e][i].shape(0); x++) {
                for (u32_t y = 0; y < dogs[e][i].shape(1); y++) {
                    auto leftUpCorner = vigra::Shape2(x - 1, y - 1);
                    auto rightDownCorner = vigra::Shape2(x + 1, y + 1);

                    //Get the neighborhood of the current pixel
                    auto current = dogs[e][i].subarray(leftUpCorner, rightDownCorner);
                    //Get neighborhood of adjacent DOGs
                    auto under = dogs[e][i - 1].subarray(leftUpCorner, rightDownCorner);
                    auto above = dogs[e][i + 1].subarray(leftUpCorner, rightDownCorner);
                    /*
                    * Check all neighborhood pixels of current and adjacent DOGs. If there isn't any
                    * pixel bigger or smaller than the current. We found an extremum.
                    */
                    if ((!any(current > dogs[e][i](x, y)) &&
                        !any(under > dogs[e][i](x, y)) &&
                        !any(above > dogs[e][i](x, y))) ||
                        (!any(current < dogs[e][i](x, y)) &&
                        !any(under < dogs[e][i](x, y)) &&
                        !any(above < dogs[e][i](x, y)))) {

                       interestPoints[e].emplace_back(std::make_tuple(x, y));
                    }
                }
            }
        }
    }
    return interestPoints;
}

const std::vector<std::vector<vigra::MultiArray<2, f32_t>>> Sift::_createDOGs(
    vigra::MultiArray<2, f32_t>& img, u16_t epochs, f32_t sigma, f32_t k, u16_t dogPerEpoch) const {

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
    gaussians[0].emplace_back(_convolveWithGauss(img, sigma));

    //TODO: More elegant way?
    u32_t exp = 0;
    for (u32_t i = 0; i < epochs; i++) {
        laplacians.emplace_back(std::vector<vigra::MultiArray<2, f32_t>>());
        for (u32_t j = 1; j < dogPerEpoch + 1; j++) {
            gaussians[i].emplace_back(_convolveWithGauss(gaussians[i][j - 1],
                std::pow(k, exp) * sigma));
            laplacians[i].emplace_back(_DOG(gaussians[i][j - 1], gaussians[i][j]));
            exp++;
        }
        /*
        * If we aren't in the last epoch populate the next level with the second
        * last element, scaled won by a half, of the gaussians.
        */
        if (i < epochs - 1) {
            gaussians.emplace_back(std::vector<vigra::MultiArray<2, f32_t>>());
            auto scaledElem = _reduceToNextLevel(
                gaussians[i][dogPerEpoch - 1], sigma);
            gaussians[i + 1].emplace_back(scaledElem);

            exp -= 2;
        }
    }

    //Save laplacianOfGaussian for Demonstration purposes
    // for (u32_t i = 0; i < laplacians.size(); i++) {
    //     for (u32_t j = 0; j < laplacians[i].size(); j++) {
    //     std::string fnStr = "images/laplacian" + std::to_string(i) + std::to_string(j)
    //         + ".png";
    //     exportImage(laplacians[i][j], vigra::ImageExportInfo(fnStr.c_str()));
    //     }
    // }
    return laplacians;
}

const vigra::MultiArray<2, f32_t> Sift::_reduceToNextLevel(const vigra::MultiArray<2, f32_t>& in,
    f32_t sigma) const {

    // image size at current level
    const i32_t height = in.height();
    const i32_t width = in.width();

    // image size at next smaller level
    const i32_t newheight = (height + 1) / 2;
    const i32_t newwidth = (width + 1) / 2;

    // resize result image to appropriate size
    vigra::MultiArray<2, f32_t> out(vigra::Shape2(newwidth, newheight));

    // downsample smoothed image
    resizeImageNoInterpolation(_convolveWithGauss(in, sigma), out);
    return out;
}

const vigra::MultiArray<2, f32_t> Sift::_convolveWithGauss(const vigra::MultiArray<2, f32_t>& input,
    f32_t sigma) const {

    vigra::Kernel1D<f64_t> filter;
    filter.initGaussian(sigma);

    vigra::MultiArray<2, f32_t> tmp(input.shape());
    vigra::MultiArray<2, f32_t> result(input.shape());

    separableConvolveX(input, tmp, filter);
    separableConvolveY(tmp, result, filter);

    return result;
}

const vigra::MultiArray<2, f32_t> Sift::_DOG(const vigra::MultiArray<2, f32_t>& lower,
    const vigra::MultiArray<2, f32_t>& higher) const {

    return higher - lower;
}
