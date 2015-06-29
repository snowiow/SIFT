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
#include <vigra/tinyvector.hxx>

#include "point.hpp"

using namespace vigra::multi_math;

void Sift::calculate(vigra::MultiArray<2, f32_t>& img) 
{
    auto dogs = _createDOGs(img);
    //Save DoGs for Demonstration purposes
    for (u16_t i = 0; i < dogs.width(); i++) {
        for (u16_t j = 0; j < dogs.height(); j++) {
            const std::string fnStr = "images/dog" + std::to_string(i) + std::to_string(j) + ".png";
            exportImage(dogs(i, j), vigra::ImageExportInfo(fnStr.c_str()));
        }
    }

    auto interestPoints = _findScaleSpaceExtrema(dogs);
    //Save img with found interest points for demonstration purposes
    auto img_output1 = img;
    for (u16_t i = 0; i < interestPoints.height(); i++) {
        for(u16_t x = 0; x < img.width(); x++) {
            for(u16_t y = 0; y < img.height(); y++) {
                if (interestPoints(0, i)(x, y) > -1) {
                    img_output1(x, y) = 255;
                }
            }
        }
    }

    exportImage(img_output1, vigra::ImageExportInfo("images/interest_points.png"));
    _eliminateEdgeResponses(interestPoints, dogs);
    //Save img with filtered interest points for demonstration purposes
    auto img_output2 = img;
    for (u16_t i = 0; i < interestPoints.height(); i++) {
        for(u16_t x = 0; x < img.width(); x++) {
            for(u16_t y = 0; y < img.height(); y++) {
                if (interestPoints(0, i)(x, y) > -1) {
                    img_output2(x, y) = 255;
                }
            }
        }
    }
    exportImage(img_output2, vigra::ImageExportInfo("images/after_filter.png"));
}


void _orientationAssignment(Matrix<Matrix<f32_t>> interestPoints) {
    //for (u16_t e = 0; e < interestPoints.width(); e++) {
        //for (u16_t i = 0; i < interestPoints.height(); i++) {
            //auto bottomLaplacianScale = _calculateScale(e, i, _sigma, _k, _dogsPerEpoch); 
            //auto topLaplacianScale = 
        //}
    //}
}

f32_t Sift::_calculateScale(u16_t epoch, u16_t index, f32_t sigma, f32_t k, u16_t dogsPerEpoch) const {
    if (epoch == 0) {
        return std::pow(k, index) * sigma;
    }
    return (epoch * (dogsPerEpoch - 2) + index) * sigma;
}

void Sift::_eliminateEdgeResponses(Matrix<Matrix<f32_t>>& interestPoints, const Matrix<vigra::MultiArray<2, f32_t>>& dogs) const {
    for(u16_t e = 0; e < dogs.width(); e++) {
        for (u16_t i = 1; i < dogs.height() - 1; i++) {
            for (u16_t x = 1; x < dogs(e, i).shape(0) - 1; x++) {
                for (u16_t y = 1; y < dogs(e, i).shape(1) - 1; y++) {
                    if (interestPoints(e, i - 1)(x, y) > -1) {
                        auto d = dogs(e, i);

                        const vigra::MultiArray<2, f32_t> param[3] = {dogs(e, i - 1), dogs(e, i), dogs(e, i + 1)};
                        const Point p(x, y);
                        auto deriv = _foDerivative(param, p);
                        auto sec_deriv = _soDerivative(param, p);

                        auto neg_sec_deriv = sec_deriv;
                        for (auto elem : neg_sec_deriv) {
                            elem *= -1;
                        }

                        auto extremum =  vigra::linalg::operator*(vigra::linalg::inverse(sec_deriv), deriv); 

                        //Calculated up 0.5 from paper to own image values [0,255]
                        if (extremum[0] > 127.5 || extremum[1] > 127.5 || extremum[2] > 127.5) {
                            interestPoints(e, i - 1)(x, y) = -1;
                            continue;
                        } 

                        auto func_val_extremum = vigra::operator*(deriv, extremum);
                        func_val_extremum *= 0.5;
                        func_val_extremum[0] += d(x,y);
                        func_val_extremum[1] += d(x,y);
                        func_val_extremum[2] += d(x,y);
                        //Calculated up 0.03 from paper to own image values[0, 255]
                        if (func_val_extremum[0] + func_val_extremum[1] + func_val_extremum[2] < 7.65) {
                            interestPoints(e, i - 1)(x, y) = -1;
                            continue;
                        }

                        //dxx + dyy
                        f32_t hessian_tr = sec_deriv(0, 0) + sec_deriv(1, 1);
                        //dxx * dyy - dxy^2
                        f32_t hessian_det = sec_deriv(0, 0) * sec_deriv(1, 1) - std::pow(sec_deriv(1, 0), 2);

                        if (hessian_det < 0) {
                            interestPoints(e, i - 1)(x, y) = -1;
                            continue;
                        }

                        if (std::pow(hessian_tr, 2) / hessian_det > std::pow(2550 + 1, 2) / 2550) {
                            interestPoints(e, i - 1)(x, y) = -1;
                        }
                    }
                }
            }
        }
    }
}

const vigra::TinyVector<f32_t, 3> Sift::_foDerivative(const vigra::MultiArray<2, f32_t> img[3], 
        const Point& p) const {

    f32_t dx = (img[1](p.x - 1, p.y) - img[1](p.x + 1, p.y)) / 2;
    f32_t dy = (img[1](p.x, p.y - 1) - img[1](p.x, p.y + 1)) / 2;
    f32_t ds = (img[0](p.x, p.y) - img[2](p.x, p.y)) / 2;

    return vigra::TinyVector<f32_t, 3>(dx, dy, ds);
}

const vigra::MultiArray<2, f32_t> Sift::_soDerivative(const vigra::MultiArray<2, f32_t> img[3], 
        const Point& p) const {

    f32_t dxx = img[1](p.x + 1, p.y) + img[1](p.x - 1, p.y) - 2 * img[1](p.x, p.y);
    f32_t dyy = img[1](p.x, p.y + 1) + img[1](p.x, p.y - 1) - 2 * img[1](p.x, p.y);
    f32_t dss = img[2](p.x, p.y) + img[0](p.x, p.y) - 2 * img[1](p.x, p.y);
    f32_t dxy = img[1](p.x + 1, p.y + 1) - img[1](p.x - 1, p.y + 1) - img[1](p.x + 1, p.y - 1) 
        + img[1](p.x - 1, p.y - 1);

    f32_t dxs = img[2](p.x + 1, p.y) - img[2](p.x - 1, p.y) 
        - img[0](p.x + 1, p.y) + img[0](p.x - 1, p.y);

    f32_t dys = img[2](p.x, p.y + 1) - img[2](p.x, p.y + 1)
        - img[0](p.x, p.y + 1) + img[0](p.x, p.y - 1);
    vigra::MultiArray<2, f32_t> sec_deriv(vigra::Shape2(3, 3));

    sec_deriv(0, 0) = dxx;
    sec_deriv(0, 1) = dxy;
    sec_deriv(0, 2) = dxs;
    sec_deriv(1, 0) = dxy;
    sec_deriv(1, 1) = dyy;
    sec_deriv(1, 2) = dys;
    sec_deriv(2, 0) = dxs;
    sec_deriv(2, 1) = dys;
    sec_deriv(2, 2) = dss;

    return sec_deriv;
}

const Matrix<Matrix<f32_t>> Sift::_findScaleSpaceExtrema(const Matrix<vigra::MultiArray<2, f32_t>>& dogs) const {
    //A matrix of matrix. Outer dogs will be ignored, because we need a upper and lower neighbor
    Matrix<Matrix<f32_t>> interestPoints(dogs.width(), dogs.height() - 2);
    for (u16_t e = 0; e < dogs.width(); e++) {
        for (u16_t i = 1; i < dogs.height() - 1; i++) {
            interestPoints(e, i - 1) = Matrix<f32_t>(dogs(e, i).width(), dogs(e, i).height(), -1);
            for (i16_t x = 0; x < dogs(e, i).shape(0); x++) {
                for (i16_t y = 0; y < dogs(e, i).shape(1); y++) {
                    auto leftUpCorner = vigra::Shape2(x - 1, y - 1);
                    auto rightDownCorner = vigra::Shape2(x + 1, y + 1);

                    //Get the neighborhood of the current pixel
                    auto current = dogs(e, i).subarray(leftUpCorner, rightDownCorner);
                    //Get neighborhood of adjacent DOGs
                    auto under = dogs(e, i - 1).subarray(leftUpCorner, rightDownCorner);
                    auto above = dogs(e, i + 1).subarray(leftUpCorner, rightDownCorner);
                    //Check all neighborhood pixels of current and adjacent DOGs. If there isn't any
                    //pixel bigger or smaller than the current. We found an extremum.
                    if ((!any(current > dogs(e, i)(x, y)) &&
                                !any(under > dogs(e, i)(x, y)) &&
                                !any(above > dogs(e, i)(x, y))) ||
                            (!any(current < dogs(e, i)(x, y)) &&
                             !any(under < dogs(e, i)(x, y)) &&
                             !any(above < dogs(e, i)(x, y))))
                    {
                        interestPoints(e, i - 1)(x, y) = dogs(e, i)(x, y);
                    }
                }
            }
        }
    }
    return interestPoints; // TODO: by ref entgegen nehmen, um copy zu vermeiden?
}

const Matrix<vigra::MultiArray<2, f32_t>> Sift::_createDOGs(vigra::MultiArray<2, f32_t>& img) {

    assert(_epochs > 0); // pre condition
    assert(_dogsPerEpoch >= 3); // pre condition

    Matrix<vigra::MultiArray<2, f32_t>> gaussians(_epochs, _dogsPerEpoch + 2);
    Matrix<vigra::MultiArray<2, f32_t>> dogs(_epochs, _dogsPerEpoch);

    gaussians(0, 0) = _convolveWithGauss(img, _sigma);

    //TODO: More elegant way?
    u16_t exp = 0;
    for (i16_t i = 0; i < _epochs; i++) {
        for (i16_t j = 1; j < _dogsPerEpoch + 1; j++) {
            gaussians(i, j) = _convolveWithGauss(gaussians(i, j - 1), std::pow(_k, exp) * _sigma);
            dogs(i, j - 1) = _dog(gaussians(i, j - 1), gaussians(i, j));
            exp++;
        }
        /*
         * If we aren't in the last epoch populate the next level with the second
         * last element, scaled by a half, of images of current epoch.
         */
        if (i < (_epochs - 1)) {
            auto scaledElem = _reduceToNextLevel(gaussians(i, _dogsPerEpoch - 1), _sigma);
            gaussians(i + 1, 0) = scaledElem;

            exp -= 2;
        }
    }
    _gaussians = gaussians;
    return dogs; // TODO: by ref entgegen nehmen um copy zu vermeiden?
}

const vigra::MultiArray<2, f32_t> Sift::_reduceToNextLevel(const vigra::MultiArray<2, f32_t>& in, 
        f32_t sigma) const {

    // image size at current level
    const vigra::Shape2 s((in.width()+ 1) / 2, (in.height() + 1) / 2);

    // resize result image to appropriate size
    vigra::MultiArray<2, f32_t> out(s);
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

const vigra::MultiArray<2, f32_t> Sift::_dog(const vigra::MultiArray<2, f32_t>& lower, 
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
