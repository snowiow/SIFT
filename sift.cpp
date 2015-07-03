#include "sift.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <cassert>
#include <exception>

#include <vigra/convolution.hxx>
#include <vigra/resizeimage.hxx>
#include <vigra/impex.hxx>
#include <vigra/multi_math.hxx>
#include <vigra/linear_algebra.hxx>
#include <vigra/tinyvector.hxx>

#include "point.hpp"

using namespace vigra::multi_math;
using namespace vigra::linalg;

void Sift::calculate(vigra::MultiArray<2, f32_t>& img) {
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
        for (u16_t x = 0; x < img.width(); x++) {
            for (u16_t y = 0; y < img.height(); y++) {
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
        for (u16_t x = 0; x < img.width(); x++) {
            for (u16_t y = 0; y < img.height(); y++) {
                if (interestPoints(0, i)(x, y) > -1) {
                    img_output2(x, y) = 255;
                }
            }
        }
    }
    exportImage(img_output2, vigra::ImageExportInfo("images/after_filter.png"));
    _orientationAssignment(interestPoints);
}


void Sift::_orientationAssignment(FMatrix<FMatrix<f32_t>> interestPoints) {
    for (u16_t e = 0; e < 1; e++) {
        for (u16_t i = 0; i < interestPoints.height(); i++) {
            auto closest = _gaussians(e, i);
            f32_t scale = _calculateScale(e, i);
            vigra::Shape2 size(closest.width(), closest.height());
            vigra::MultiArray<2, f32_t> magnitudes(size);
            vigra::MultiArray<2, f32_t> orientations(size);
            for (u16_t x = 0; x < 1; x++) {
                for (u16_t y = 0; y < closest.height(); y++) {
                    Point p(x, y);
                    magnitudes(x, y) = _gradientMagnitude(closest, p);
                    orientations(x, y) = _gradientOrientation(closest, p);
                }
            }
            for (u16_t x = 8; x < interestPoints(e, i).width() - 8; x++) {
                for (u16_t y = 8; y < interestPoints(e, i).height() - 8; y++) {
                    if (interestPoints(e, i)(x, y) >= 0) {
                        auto topLeftCorner = vigra::Shape2(x - 8, y - 8);
                        auto bottomRightCorner = vigra::Shape2(x + 8, y + 8);

                        auto orientation_region = orientations.subarray(topLeftCorner, bottomRightCorner);
                        auto mag_region = magnitudes.subarray(topLeftCorner, bottomRightCorner);

                        auto histogram = _orientationHistogram(orientation_region, mag_region, scale);

                        _createPeak(histogram);

                        //TODO:: Parabola to the 3 values closest to the peak

                    }
                }
            }
        }
    }
}

void Sift::_createPeak(std::array<f32_t, 36>& histo) {
    const f32_t max = *(std::max_element(histo.begin(), histo.end()));
    //allowed range(80% of max)
    const f32_t range = max / 5 * 4;
    std::for_each(histo.begin(), histo.end(), [&](f32_t& elem) { if (elem < range) elem = -1; });
}

f32_t Sift::_calculateScale(u16_t e, u16_t i) const {
    if (e < 1) {
        if (i < 1)  {
            return _sigma;
        }   
        return std::pow(_k, i) * _sigma;
    } else if (e == 1) {
        return std::pow(_k, _dogsPerEpoch + i) * _sigma;
    }
    //_dogsPerEpoch + 2: count of gaussians per epoch
    //e - 1: the very last epoch before our
    // - 2: take the top img from the stack and add i to it
    return std::pow(_k, (_dogsPerEpoch + 2) * (e - 1) - 2 + i) * _sigma;
}

const std::array<f32_t, 36> Sift::_orientationHistogram(const vigra::MultiArray<2, f32_t>& orientations,
        const vigra::MultiArray<2, f32_t>& magnitudes, f32_t scale) const {

    std::array<f32_t, 36> bins;
    vigra::MultiArray<2, f32_t> gauss_mag = _convolveWithGauss(magnitudes, 1.5 * scale);
    for (u16_t x = 0; x < orientations.width(); x++) {
        for (u16_t y = 0; y < orientations.height(); y++) {
            u16_t i = std::floor(orientations(x, y) / 10);
                //TODO: Don't know if it can happen, that a orientation is exactly 360°. If not, delete this.
                i = i > 35 ? 35 : i;
            bins[i] += gauss_mag(x, y);
        }
    }
    return bins;
}

f32_t Sift::_gradientMagnitude(const vigra::MultiArray<2, f32_t>& img, const Point& p) const {
    return std::sqrt(std::pow(img(p.x + 1, p.y) - img(p.x - 1, p.y), 2) + 
            std::pow(img(p.x, p.y + 1) - img(p.x, p.y - 1), 2));
}

f32_t Sift::_gradientOrientation(const vigra::MultiArray<2, f32_t>& img, const Point& p) const {
    return std::atan2(img(p.x, p.y + 1) - img(p.x, p.y - 1), img(p.x + 1, p.y) - img(p.x - 1, p.y));
}

void Sift::_eliminateEdgeResponses(FMatrix<FMatrix<f32_t>>& interestPoints, 
        const FMatrix<vigra::MultiArray<2, f32_t>>& dogs) const {

    for(u16_t e = 0; e < dogs.width(); e++) {
        for (u16_t i = 1; i < dogs.height() - 1; i++) {
            for (u16_t x = 1; x < dogs(e, i).shape(0) - 1; x++) {
                for (u16_t y = 1; y < dogs(e, i).shape(1) - 1; y++) {
                    if (interestPoints(e, i - 1)(x, y) > -1) {
                        auto d = dogs(e, i);

                        const vigra::MultiArray<2, f32_t> param[3] = {dogs(e, i - 1), dogs(e, i), dogs(e, i + 1)};
                        const Point p(x, y);
                        vigra::Matrix<f32_t> deriv = _foDerivative(param, p);
                        vigra::Matrix<f32_t> sec_deriv = _soDerivative(param, p);

                        vigra::Matrix<f32_t> neg_sec_deriv = sec_deriv ;
                        neg_sec_deriv *=  -1;

                        vigra::MultiArray<2, f32_t> extremum(vigra::Shape2(3, 1));
                        if (!linearSolve(inverse(neg_sec_deriv), deriv, extremum)) {
                            std::cerr << "Couldn't solve linear system" << std::endl;
                            throw;
                        }

                        //Calculated up 0.5 from paper to own image values [0,255]
                        if (extremum(0, 0) > 127.5 || extremum(1, 0) > 127.5 || extremum(2, 0) > 127.5) {
                            interestPoints(e, i - 1)(x, y) = -1;
                            continue;
                        } 
                        vigra::Matrix<f32_t> deriv_transpose = deriv.transpose();
                        f32_t func_val_extremum = dot(deriv_transpose, extremum);
                        func_val_extremum *= 0.5 + d(x, y);

                        //Calculated up 0.03 from paper to own image values[0, 255]
                        if (func_val_extremum < 7.65) {
                            interestPoints(e, i - 1)(x, y) = -1;
                            continue;
                        }

                        //dxx + dyy
                        f32_t hessian_tr = sec_deriv(0, 0) + sec_deriv(1, 1);
                        //dxx * dyy - dxy^2
                        f32_t hessian_det = sec_deriv(0, 0) * sec_deriv(1, 1) - std::pow(sec_deriv(0, 1), 2);

                        if (hessian_det < 0) {
                            interestPoints(e, i - 1)(x, y) = -1;
                            continue;
                        }

                        //Original r = 10, calculated up to own image values[0, 255]
                        if (std::pow(hessian_tr, 2) / hessian_det > std::pow(2550 + 1, 2) / 2550) {
                            interestPoints(e, i - 1)(x, y) = -1;
                        }
                    }
                }
            }
        }
    }
}

const vigra::Matrix<f32_t> Sift::_foDerivative(const vigra::MultiArray<2, f32_t> img[3], 
        const Point& p) const {

    f32_t dx = (img[1](p.x - 1, p.y) - img[1](p.x + 1, p.y)) / 2;
    f32_t dy = (img[1](p.x, p.y - 1) - img[1](p.x, p.y + 1)) / 2;
    f32_t ds = (img[0](p.x, p.y) - img[2](p.x, p.y)) / 2;
    vigra::Matrix<f32_t> result(vigra::Shape2(3, 1));
    result(0, 0) = dx;
    result(1, 0) = dy;
    result(2, 0) = ds;
    return result;
}

const vigra::Matrix<f32_t> Sift::_soDerivative(const vigra::MultiArray<2, f32_t> img[3], 
        const Point& p) const {

    f32_t dxx = img[1](p.x + 1, p.y) + img[1](p.x - 1, p.y) - 2 * img[1](p.x, p.y);
    f32_t dyy = img[1](p.x, p.y + 1) + img[1](p.x, p.y - 1) - 2 * img[1](p.x, p.y);
    f32_t dss = img[2](p.x, p.y) + img[0](p.x, p.y) - 2 * img[1](p.x, p.y);
    f32_t dxy = (img[1](p.x + 1, p.y + 1) - img[1](p.x - 1, p.y + 1) - img[1](p.x + 1, p.y - 1) 
        + img[1](p.x - 1, p.y - 1)) / 2;

    f32_t dxs = (img[2](p.x + 1, p.y) - img[2](p.x - 1, p.y) 
        - img[0](p.x + 1, p.y) + img[0](p.x - 1, p.y)) / 2;

    f32_t dys = (img[2](p.x, p.y + 1) - img[2](p.x, p.y + 1)
        - img[0](p.x, p.y + 1) + img[0](p.x, p.y - 1)) / 2;
    vigra::MultiArray<2, f32_t> sec_deriv(vigra::Shape2(3, 3));

    sec_deriv(0, 0) = dxx;
    sec_deriv(1, 0) = dxy;
    sec_deriv(2, 0) = dxs;
    sec_deriv(0, 1) = dxy;
    sec_deriv(1, 1) = dyy;
    sec_deriv(2, 1) = dys;
    sec_deriv(0, 2) = dxs;
    sec_deriv(1, 2) = dys;
    sec_deriv(2, 2) = dss;

    return sec_deriv;
}

const FMatrix<FMatrix<f32_t>> Sift::_findScaleSpaceExtrema(const FMatrix<vigra::MultiArray<2, f32_t>>& dogs) const {
    //A matrix of matrix. Outer dogs will be ignored, because we need a upper and lower neighbor
    FMatrix<FMatrix<f32_t>> interestPoints(dogs.width(), dogs.height() - 2);
    for (u16_t e = 0; e < dogs.width(); e++) {
        for (u16_t i = 1; i < dogs.height() - 1; i++) {
            interestPoints(e, i - 1) = FMatrix<f32_t>(dogs(e, i).width(), dogs(e, i).height(), -1);
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

const FMatrix<vigra::MultiArray<2, f32_t>> Sift::_createDOGs(vigra::MultiArray<2, f32_t>& img) {

    assert(_epochs > 0); // pre condition
    assert(_dogsPerEpoch >= 3); // pre condition

    FMatrix<vigra::MultiArray<2, f32_t>> gaussians(_epochs, _dogsPerEpoch + 2);
    FMatrix<vigra::MultiArray<2, f32_t>> dogs(_epochs, _dogsPerEpoch);

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
    for (u16_t x = 0; x < lower.shape(0); x++) {
        for (u16_t y = 0; y < lower.shape(1); y++) {
            f32_t dif = higher(x, y) - lower(x, y);
            result(x, y) = 128 + dif;
        }
    }
    return result;
}
