#include "sift.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <cassert>

#include <vigra/convolution.hxx>
#include <vigra/resizeimage.hxx>
#include <vigra/impex.hxx>
#include <vigra/multi_math.hxx>

#include "point.hpp"

using namespace vigra::multi_math;

void Sift::calculate(vigra::MultiArray<2, f32_t>& img, u16_t epochs, f32_t sigma, f32_t k, 
        u16_t dogPerEpoch) const {


    auto dogs = _createDOGs(img, epochs, sigma, k, dogPerEpoch);
    auto& interestPoints = _findScaleSpaceExtrema(dogs);

    //Save img with found interest points for demonstration purposes
    auto img_output1 = img;
    for (auto& img : interestPoints[0]) {
        //std::cout << img.width() << std::endl;
        //std::cout << img.height() << std::endl;
        for(u16_t x = 0; x < img.width(); x++) {
            for(u16_t y = 0; y < img.height(); y++) {
                if (img[Point(x, y)] > -1) {
                    img_output1(x, y) = 255;
                }

            }
        }
    }

    exportImage(img_output1, vigra::ImageExportInfo("images/interest_points.png"));
    //_eliminateEdgeResponses(interestPoints, dogs);

    //Save img again without edge responses
    //auto img_output2 = img;
    //for (auto epoch : interestPoints[0]) {
    //for (auto point : epoch) {
    //if (point != -1)
    //img_output2(point) = 255;
    //}
    //}

    //exportImage(img_output2, vigra::ImageExportInfo("images/interest_points2.png"));
}

const vigra::MultiArray<2, f32_t> _differenceOfNeighbouringSampleX(
        const FlatMatrix<f32_t>& interestPoints, const vigra::MultiArray<2, f32_t>& d) {

    vigra::MultiArray<2, f32_t> derivative = d;
    //for (u32_t x = 1; x < interestPoints.width() - 1; x++) {
    //for(u32_t y = 0; y < interestPoints.height(); y++) {

    //}
    //}
    return derivative;
}

void Sift::_eliminateEdgeResponses(interest_point_epochs& interestPoints, 
        const img_epochs& dogs, u16_t r) const {

    //for (u32_t i = 0; i < interestPoints.size(); i++) {
    //for (u32_t j = 0; j < interestPoints[i].size(); j++) {
    //auto iter = interestPoints[i][j].begin();
    //while (iter != interestPoints[i][j].end()) {
    //// calculate Tr(D(x,x) + D(y, y))
    //auto coords = *iter; 
    //i32_t dxx = dogs[i][j](std::get<0>(coords), std::get<0>(coords));
    //i32_t dyy = dogs[i][j](std::get<1>(coords), std::get<1>(coords));
    //i32_t tr= dxx + dyy;

    ////calculate Det = D(x, x) * D(y, y) - D(x, y) ^ 2
    //i32_t det = dxx * dyy - std::pow(dogs[i][j](std::get<0>(coords), std::get<1>(coords)), 2);

    ////delete if determinant is negative
    //if (det < 0) {
    //iter = interestPoints[i][j].erase(iter);
    //continue;
    //}
    ////paper calculates with img values between 0 and 1. Ours are between 0 255.
    //r *= 255;
    ////Is principal curvature above the given threshold
    //if ((std::pow(tr, 2) / det) > (std::pow(r + 1, 2) / r)) {
    //iter = interestPoints[i][j].erase(iter);
    //continue;
    //}
    //++iter;
    //}
    //}
    //}
}

void Sift::_keypointLocation(interest_point_epochs& interestPoints) const {
    //?
}

const interest_point_epochs Sift::_findScaleSpaceExtrema(const img_epochs& dogs) const {
    //a Vector with epochs containing tuples of interest points found per epoch
    interest_point_epochs interestPoints;
    for (u16_t e = 0; e < dogs.size(); e++) {
        interestPoints.emplace_back(std::vector<FlatMatrix<f32_t>>());
        for (u16_t i = 1; i < dogs[e].size() - 1; i++) {
            std::cout << dogs[e][i].shape(0) << std::endl;
            std::cout << dogs[e][i].shape(1) << std::endl;
            interestPoints[e].emplace_back(FlatMatrix<f32_t>(dogs[e][i].shape(0), dogs[e][i].shape(1), -1));
            for (i32_t x = 0; x < dogs[e][i].shape(0); x++) {
                for (i32_t y = 0; y < dogs[e][i].shape(1); y++) {
                    auto leftUpCorner = vigra::Shape2(x - 1, y - 1);
                    auto rightDownCorner = vigra::Shape2(x + 1, y + 1);

                    //Get the neighborhood of the current pixel
                    auto current = dogs[e][i].subarray(leftUpCorner, rightDownCorner);
                    //Get neighborhood of adjacent DOGs
                    auto under = dogs[e][i - 1].subarray(leftUpCorner, rightDownCorner);
                    auto above = dogs[e][i + 1].subarray(leftUpCorner, rightDownCorner);
                    //Check all neighborhood pixels of current and adjacent DOGs. If there isn't any
                    //pixel bigger or smaller than the current. We found an extremum.
                    if ((!any(current > dogs[e][i](x, y)) &&
                                !any(under > dogs[e][i](x, y)) &&
                                !any(above > dogs[e][i](x, y))) ||
                            (!any(current < dogs[e][i](x, y)) &&
                             !any(under < dogs[e][i](x, y)) &&
                             !any(above < dogs[e][i](x, y))))
                    {
                        interestPoints[e].back()[Point(x, y)] = dogs[e][i](x, y);
                    }
                    else
                        interestPoints[e].back()[Point(x, y)] = -1;
                }
            }
        }
    }
    return interestPoints; // TODO: by ref entgegen nehmen, um copy zu vermeiden?
}

const img_epochs Sift::_createDOGs(vigra::MultiArray<2, f32_t>& img, u16_t epochs, f32_t sigma, 
        f32_t k, u16_t dogPerEpoch) const {

    assert(epochs > 0); // pre condition
    assert(dogPerEpoch >= 3); // pre condition

    /*
     * A vector which contains a vector for each epoch. Those are filled with the
     * Images which were created from the laplacianOfGaussians.
     */
    img_epochs laplacians;
    /*
     * a vector which contains a vector for each epoch. Those are filled with the
     * images which were created from gaussians
     */
    img_epochs gaussians;

    //add inital image
    gaussians.emplace_back(std::vector<vigra::MultiArray<2, f32_t>>());
    gaussians[0].emplace_back(_convolveWithGauss(img, sigma));

    //TODO: More elegant way?
    u16_t exp = 0;
    for (i32_t i = 0; i < epochs; i++) {
        laplacians.emplace_back(std::vector<vigra::MultiArray<2, f32_t>>());
        for (i32_t j = 1; j < dogPerEpoch + 1; j++) {
            gaussians[i].emplace_back(_convolveWithGauss(gaussians[i][j - 1], std::pow(k, exp) * sigma));
            laplacians[i].emplace_back(_DOG(gaussians[i][j - 1], gaussians[i][j]));
            exp++;
        }
        /*
         * If we aren't in the last epoch populate the next level with the second
         * last element, scaled won by a half, of the gaussians.
         */
        if (i < (epochs - 1)) {
            gaussians.emplace_back(std::vector<vigra::MultiArray<2, f32_t>>());
            auto scaledElem = _reduceToNextLevel(gaussians[i][dogPerEpoch - 1], sigma);
            gaussians[i + 1].emplace_back(scaledElem);

            exp -= 2;
        }
    }

    //Save laplacianOfGaussian for Demonstration purposes
    for (u16_t i = 0; i < laplacians.size(); i++) {
        for (u16_t j = 0; j < laplacians[i].size(); j++) {
            const std::string fnStr = "images/laplacian" + std::to_string(i) + std::to_string(j) + ".png";
            exportImage(laplacians[i][j], vigra::ImageExportInfo(fnStr.c_str()));
        }
    }

    return laplacians; // TODO: by ref entgegen nehmen um copy zu vermeiden?
}

const vigra::MultiArray<2, f32_t> Sift::_reduceToNextLevel(const vigra::MultiArray<2, f32_t>& in, 
        f32_t sigma) const {
    // image size at current level
    const u16_t height = in.height();
    const u16_t width = in.width();

    // image size at next smaller level
    const u16_t newheight = (height + 1) / 2;
    const u16_t newwidth = (width + 1) / 2;

    // resize result image to appropriate size
    vigra::MultiArray<2, f32_t> out(vigra::Shape2(newwidth, newheight));
    // downsample smoothed image
    resizeImageNoInterpolation(_convolveWithGauss(in, sigma), out);

    return out; // TODO by ref entgegen nehmen um copy zu vermeiden?
}

const vigra::MultiArray<2, f32_t> Sift::_convolveWithGauss(const vigra::MultiArray<2, f32_t>& input, 
        f32_t sigma) const {

    vigra::Kernel1D<f32_t> filter;
    filter.initGaussian(sigma);
    vigra::MultiArray<2, f32_t> tmp(input.shape());
    vigra::MultiArray<2, f32_t> result(input.shape());

    separableConvolveX(input, tmp, filter);
    separableConvolveY(tmp, result, filter);

    return result;
}

const vigra::MultiArray<2, f32_t> Sift::_DOG(const vigra::MultiArray<2, f32_t>& lower, 
        const vigra::MultiArray<2, f32_t>& higher) const {

    vigra::MultiArray<2, f32_t> result(vigra::Shape2(lower.shape()));
    for(u16_t x = 0; x < lower.shape(0); x++) {
        for(u16_t y = 0; y < lower.shape(1); y++) {
            f32_t dif = higher(x, y) - lower(x, y);
            result(x, y) = 128 + dif;
        }
    }
    return result;
}
