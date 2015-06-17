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
#include <vigra/linear_algebra.hxx>

#include "point.hpp"

using namespace vigra::multi_math;

void Sift::calculate(
    vigra::MultiArray<2, f32_t>& img, u16_t epochs, f32_t sigma, f32_t k, u16_t dogPerEpoch) const
{
    auto dogs = _createDOGs(img, epochs, sigma, k, dogPerEpoch);
    auto interestPoints = _findScaleSpaceExtrema(dogs);
    //Save img with found interest points for demonstration purposes
    auto img_output1 = img;
    for (auto& img : interestPoints[0]) {
        for(u16_t x = 0; x < img.width(); x++) {
            for(u16_t y = 0; y < img.height(); y++) {
                if (img[Point(x, y)] > -1) {
                    img_output1(x, y) = 255;
                }
            }
        }
    }

    exportImage(img_output1, vigra::ImageExportInfo("images/interest_points.png"));
    /*
    _keypointLocation(interestPoints, dogs);
    
    auto img_output2 = img;
    for (auto& img : interestPoints[0]) {
        for(u16_t x = 0; x < img.width(); x++) {
            for(u16_t y = 0; y < img.height(); y++) {
                if (img[Point(x, y)] > -1) {
                    img_output2(x, y) = 255;
                }
            }
        }
    }
    
    exportImage(img_output2, vigra::ImageExportInfo("images/after_keypointLocation.png"));
    */
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

const vigra::MultiArray<2, f32_t> Sift::_differenceOfNeighbouringSampleX(
        const Matrix<f32_t>& interestPoints, const vigra::MultiArray<2, f32_t>& d) const {
    vigra::MultiArray<2, f32_t> derivative = d;
    for (u16_t x = 1; x < d.width() - 1; x++) {
        for(u16_t y = 0; y < d.height(); y++) {
            if (interestPoints(x - 1, y) > -1 || interestPoints(x + 1, y) > -1) {
                derivative(x, y) = std::abs(d(x - 1, y) - d(x + 1, y));
            }
        }
    }
    return derivative;
}

const vigra::MultiArray<2, f32_t> Sift::_differenceOfNeighbouringSampleY(
        const Matrix<f32_t>& interestPoints, const vigra::MultiArray<2, f32_t>& d) const {
    vigra::MultiArray<2, f32_t> derivative = d;
    for (u16_t x = 0; x < d.width(); x++) {
        for(u16_t y = 1; y < d.height() - 1; y++) {
            if (interestPoints(x, y - 1) > -1 || interestPoints(x, y + 1) > -1) {
                derivative(x, y) = std::abs(d(x, y - 1) - d(x, y + 1));
            }
        }
    }
    return derivative;
}

const vigra::MultiArray<2, f32_t> Sift::_differenceOfNeighbouringSampleSigma(
            const std::vector<Matrix<f32_t>>& interestPoints, 
            const std::vector<vigra::MultiArray<2, f32_t>>& d) const {
    assert(interestPoints.size() == 3);
    assert(d.size() == 3);
    vigra::MultiArray<2, f32_t> derivative = d[1];
    
    for (u16_t x = 0; x  < derivative.width(); x++) {
        for (u16_t y = 0; y < derivative.height(); y++) {
           if (interestPoints[0](x, y) > -1 && interestPoints[2](x, y) > -1)  {
                derivative(x, y) = std::abs(d[0](x, y) - d[2](x, y));
           }
        }
    }
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

void Sift::_keypointLocation(interest_point_epochs& interestPoints, const img_epochs& dogs) const {
    for(u16_t e = 0; e < dogs.size(); e++) {
        for(u16_t i = 1; i < dogs[e].size() - 1; i++) {
            auto dx = _differenceOfNeighbouringSampleX(interestPoints[e][i], dogs[e][i]);
            auto dxx = _differenceOfNeighbouringSampleX(interestPoints[e][i], dx);

            auto dy = _differenceOfNeighbouringSampleY(interestPoints[e][i], dogs[e][i]);
            auto dyy = _differenceOfNeighbouringSampleY(interestPoints[e][i], dy);

            auto ds = _differenceOfNeighbouringSampleSigma(interestPoints[e], dogs[e]);
            std::vector<vigra::MultiArray<2, f32_t>> v;

            v.emplace_back(dogs[e][i - 1]);
            v.emplace_back(ds);
            v.emplace_back(dogs[e][i + 1]);
            auto dss = _differenceOfNeighbouringSampleSigma(interestPoints[e], v);

            vigra::MultiArray<1, vigra::MultiArray<2, f32_t>> v1(vigra::Shape1(3));
            v1[0] = dx;
            v1[1] = dy;
            v1[2] = ds;
            vigra::MultiArray<1, vigra::MultiArray<2, f32_t>> v2(vigra::Shape1(3));
            v2[0] = dxx;
            v2[1] = dyy;
            v2[2] = dss;
            auto extremum = v1 * v2;
            v1[0] = vigra::linalg::transpose(v1[0]);
            v1[1] = vigra::linalg::transpose(v1[1]);
            v1[2] = vigra::linalg::transpose(v1[2]);
            //vigra::MultiArray<2, f32_t> ext_sub = dogs[e][i] + 0.5 * v1 * extremum;

            //for (u16_t x = 1; x < ext_sub.shape(0) - 1; x++) {
                //for (u16_t y = 1; y < ext_sub.shape(1) - 1; y++) {
                    //if (ext_sub(x, y) / 255 > 0.03) {
                        //interestPoints[e][i](x, y) = -1;
                    //}
                //}
            //}
        }
    }
}

const interest_point_epochs Sift::_findScaleSpaceExtrema(const img_epochs& dogs) const {
    //a Vector with epochs containing tuples of interest points found per epoch
    interest_point_epochs interestPoints;
    for (u16_t e = 0; e < dogs.size(); e++) {
        interestPoints.emplace_back(std::vector<Matrix<f32_t>>());
        for (u16_t i = 1; i < dogs[e].size() - 1; i++) {
            interestPoints[e].emplace_back(Matrix<f32_t>(dogs[e][i].shape(0), dogs[e][i].shape(1), -1));
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
