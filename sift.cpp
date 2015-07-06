#include "sift.hpp"

#include <iostream>
#include <string>
#include <cassert>

#include <vigra/impex.hxx>
#include <vigra/multi_math.hxx>
#include <vigra/linear_algebra.hxx>

#include "point.hpp"
#include "algorithms.hpp"

using namespace vigra::multi_math;
using namespace vigra::linalg;

namespace sift {
    void Sift::calculate(vigra::MultiArray<2, f32_t>& img) {
        auto dogs = _createDOGs(img);
        //Save DoGs for Demonstration purposes
        for (u16_t i = 0; i < dogs.width(); i++) {
            for (u16_t j = 0; j < dogs.height(); j++) {
                const std::string fnStr = "images/dog" + std::to_string(i) + std::to_string(j) + ".png";
                exportImage(dogs(i, j).img, vigra::ImageExportInfo(fnStr.c_str()));
            }
        }

        std::vector<InterestPoint> interestPoints = _findScaleSpaceExtrema(dogs);
        //Save img with found interest points for demonstration purposes
        auto img_output1 = img;
        for (InterestPoint p : interestPoints) {
            img_output1(p.loc.x, p.loc.y) = 255;
        }

        exportImage(img_output1, vigra::ImageExportInfo("images/interest_points.png"));
        _eliminateEdgeResponses(interestPoints, dogs);

        //Cleanup
        std::sort(interestPoints.begin(), interestPoints.end(), InterestPoint::cmpByFilter);
        auto result = std::find_if(interestPoints.begin(), interestPoints.end(), 
                [](const InterestPoint& p) { return p.filtered; });

        u16_t size = std::distance(interestPoints.begin(), result);
        interestPoints.resize(size);

        //Save img with filtered interest points for demonstration purposes
        auto img_output2 = img;
        for (InterestPoint& p : interestPoints) {
            img_output2(p.loc.x, p.loc.y) = 255;
        }
        exportImage(img_output2, vigra::ImageExportInfo("images/after_filter.png"));
        _orientationAssignment(interestPoints);
    }


    void Sift::_orientationAssignment(std::vector<InterestPoint>& interestPoints) {
        for (InterestPoint& p : interestPoints) {
            OctaveElem closest = _findNearestGaussian(p.scale);

            u16_t region = 8;
            //Is Keypoint inside image boundaries of gaussian
            if ((p.loc.x < region || p.loc.x >= closest.img.width() - region) ||
                    (p.loc.y < region && p.loc.y >= closest.img.height() - region)) {

                p.filtered = true;
                continue;
            }

            auto topLeftCorner = vigra::Shape2(p.loc.x - region, p.loc.y - region);
            auto bottomRightCorner = vigra::Shape2(p.loc.x + region, p.loc.y + region);
            auto gauss_region = closest.img.subarray(topLeftCorner, bottomRightCorner);

            vigra::MultiArray<2, f32_t> magnitudes(vigra::Shape2(region * 2, region * 2));
            vigra::MultiArray<2, f32_t> orientations(vigra::Shape2(region * 2, region * 2));

            for (u16_t x = 0; x < magnitudes.width(); x++) {
                for (u16_t y = 0; y < magnitudes.height(); y++) {
                    Point<u16_t, u16_t> point(x, y);
                    magnitudes(x, y) = alg::gradientMagnitude(closest.img, point);
                    orientations(x, y) = alg::gradientOrientation(closest.img, point);

                }
            }
            auto gauss_convolved = alg::convolveWithGauss(gauss_region, 1.5 * p.scale);
            auto histogram = alg::orientationHistogram(orientations, magnitudes, gauss_region);
            auto peaks = _findPeaks(histogram);
        }
    }

    const OctaveElem& Sift::_findNearestGaussian(f32_t scale) {
        f32_t lowest_diff = 100;
        OctaveElem& nearest_gauss = _gaussians(0, 0);
        for (u16_t o = 0; o < _gaussians.width(); o++) {
            //TODO: Abrechen wenn scale nur noch ansteigt
            for (u16_t i = 0; i < _gaussians.height(); i++) {
                if (std::abs(_gaussians(o, i).scale - scale) < lowest_diff)
                    nearest_gauss = _gaussians(o, i);
            }
        }
        return nearest_gauss;
    }

    const std::array<f32_t, 36> Sift::_findPeaks(const std::array<f32_t, 36>& histo) const {
        auto peaks_only = histo;
        auto result_iter = std::max_element(histo.begin(), histo.end());
        u16_t  max_index = std::distance(histo.begin(), result_iter);
        while (max_index < 1 || max_index >= histo.size() - 1) {
            peaks_only[max_index] = -1;
            result_iter = std::max_element(histo.begin(), histo.end());
            max_index = std::distance(histo.begin(), result_iter);
        }
        //ignore edges, because they cant be aproximated appropiately
        peaks_only[0] = -1;
        peaks_only[peaks_only.size() - 1] = -1;

        //allowed range(80% of max) 
        const f32_t range = (*result_iter) * (4 / 5);
        std::for_each(peaks_only.begin(), peaks_only.end(), [&](f32_t& elem) { if (elem < range) elem = -1; }); 
        //aproximate peak with vertex parabola. Here we need the 360° space. +5 Because we just have
        //10° bins, so we just take the middle of the bin.
        Point<u16_t, f32_t> ln((max_index - 1) * 10 + 5, histo[max_index - 1]);
        Point<u16_t, f32_t> peak(max_index * 10 + 5, histo[max_index]);
        Point<u16_t, f32_t> rn((max_index + 1) * 10 + 5, histo[max_index + 1]);
        peaks_only[max_index] = alg::vertexParabola(ln, peak, rn);

        for (u16_t i = 0; i < peaks_only.size(); i++) {
            if (peaks_only[i] > - 1 && i != max_index) {
                Point<u16_t, f32_t> ln((i - 1) * 10 + 5, histo[i - 1]);
                Point<u16_t, f32_t> peak(i * 10 + 5, histo[i]);
                Point<u16_t, f32_t> rn((i + 1) * 10 + 5, histo[i + 1]);
                peaks_only[i] = alg::vertexParabola(ln, peak, rn);
            }
        }
        return peaks_only;
    }

    void Sift::_eliminateEdgeResponses(std::vector<InterestPoint>& interestPoints, 
            const Matrix<OctaveElem>& dogs) const {

        for(u16_t e = 0; e < dogs.width(); e++) {
            for (u16_t i = 1; i < dogs.height() - 1; i++) {
                //TODO: Hier braucht jeder Punkt nur einmal durchlaufen werden und nicht in jeder
                //Iteration erneut
                for (InterestPoint& p : interestPoints) {
                    if (p.scale == dogs(e, i).scale) {
                        auto d = dogs(e, i);
                        const vigra::MultiArray<2, f32_t> param[3] = {dogs(e, i - 1).img, dogs(e, i).img, dogs(e, i + 1).img};
                        vigra::Matrix<f32_t> deriv = alg::foDerivative(param, p.loc);
                        vigra::Matrix<f32_t> sec_deriv = alg::soDerivative(param, p.loc);

                        vigra::Matrix<f32_t> neg_sec_deriv = sec_deriv ;
                        neg_sec_deriv *=  -1;

                        vigra::MultiArray<2, f32_t> extremum(vigra::Shape2(3, 1));
                        if (!linearSolve(inverse(neg_sec_deriv), deriv, extremum)) {
                            std::cerr << "Couldn't solve linear system" << std::endl;
                            throw;
                        }

                        //Calculated up 0.5 from paper to own image values [0,255]
                        if (extremum(0, 0) > 127.5 || extremum(1, 0) > 127.5 || extremum(2, 0) > 127.5) {
                            p.filtered = true;
                            continue;
                        } 
                        vigra::Matrix<f32_t> deriv_transpose = deriv.transpose();
                        f32_t func_val_extremum = dot(deriv_transpose, extremum);
                        func_val_extremum *= 0.5 + d.img(p.loc.x, p.loc.y);

                        //Calculated up 0.03 from paper to own image values[0, 255]
                        if (func_val_extremum < 7.65) {
                            p.filtered = true;
                            continue;
                        }

                        //dxx + dyy
                        f32_t hessian_tr = sec_deriv(0, 0) + sec_deriv(1, 1);
                        //dxx * dyy - dxy^2
                        f32_t hessian_det = sec_deriv(0, 0) * sec_deriv(1, 1) - std::pow(sec_deriv(0, 1), 2);

                        if (hessian_det < 0) {
                            p.filtered = true;
                            continue;
                        }

                        //Original r = 10, calculated up to own image values[0, 255]
                        //Question: The value 10 isn't based on the greyvalues? 
                        if (std::pow(hessian_tr, 2) / hessian_det > std::pow(10 + 1, 2) / 10) 
                            p.filtered = true;
                    }
                }
            }
        }
    }

    const std::vector<InterestPoint> Sift::_findScaleSpaceExtrema(const Matrix<OctaveElem>& dogs) const {
        //A matrix of matrix. Outer dogs will be ignored, because we need a upper and lower neighbor
        std::vector<InterestPoint> interestPoints;
        for (u16_t e = 0; e < dogs.width(); e++) {
            for (u16_t i = 1; i < dogs.height() - 1; i++) {
                for (i16_t x = 1; x < dogs(e, i).img.width() - 1; x++) {
                    for (i16_t y = 1; y < dogs(e, i).img.height() - 1; y++) {
                        auto leftUpCorner = vigra::Shape2(x - 1, y - 1);
                        auto rightDownCorner = vigra::Shape2(x + 1, y + 1);

                        //Get the neighborhood of the current pixel
                        auto current = dogs(e, i).img.subarray(leftUpCorner, rightDownCorner);
                        //Get neighborhood of adjacent DOGs
                        auto under = dogs(e, i - 1).img.subarray(leftUpCorner, rightDownCorner);
                        auto above = dogs(e, i + 1).img.subarray(leftUpCorner, rightDownCorner);
                        //Check all neighborhood pixels of current and adjacent DOGs. If there isn't any
                        //pixel bigger or smaller than the current. We found an extremum.
                        if ((!any(current > dogs(e, i).img(x, y)) &&
                                    !any(under > dogs(e, i).img(x, y)) &&
                                    !any(above > dogs(e, i).img(x, y))) ||
                                (!any(current < dogs(e, i).img(x, y)) &&
                                 !any(under < dogs(e, i).img(x, y)) &&
                                 !any(above < dogs(e, i).img(x, y))))
                        {
                            interestPoints.emplace_back(InterestPoint(Point<u16_t, u16_t>(x, y), dogs(e, i).scale));
                        }
                    }
                }
            }
        }
        return interestPoints; // TODO: by ref entgegen nehmen, um copy zu vermeiden?
    }

    const Matrix<OctaveElem> Sift::_createDOGs(vigra::MultiArray<2, f32_t>& img) {
        assert(_octaves > 0); // pre condition
        assert(_dogsPerEpoch >= 3); // pre condition

        Matrix<OctaveElem> gaussians(_octaves, _dogsPerEpoch + 2);
        Matrix<OctaveElem> dogs(_octaves, _dogsPerEpoch);

        gaussians(0, 0).scale = _sigma;
        gaussians(0, 0).img = alg::convolveWithGauss(img, _sigma);

        //TODO: More elegant way?
        u16_t exp = 0;
        for (i16_t i = 0; i < _octaves; i++) {
            for (i16_t j = 1; j < _dogsPerEpoch + 1; j++) {
                f32_t scale = std::pow(_k, exp) * _sigma;
                gaussians(i, j).scale = scale;
                gaussians(i, j).img = alg::convolveWithGauss(gaussians(i, j - 1).img, scale);

                dogs(i, j - 1).scale = gaussians(i, j).scale - gaussians(i, j - 1).scale;
                dogs(i, j - 1).img = alg::dog(gaussians(i, j - 1).img, gaussians(i, j).img);
                exp++;
            }
            /*
             * If we aren't in the last octave populate the next level with the second
             * last element, scaled by a half, of images of current octave.
             */
            if (i < (_octaves - 1)) {
                auto scaledElem = alg::reduceToNextLevel(gaussians(i, _dogsPerEpoch - 1).img, 
                        gaussians(i, _dogsPerEpoch - 1).scale);
                gaussians(i + 1, 0).scale = gaussians(i, _dogsPerEpoch - 1).scale;
                gaussians(i + 1, 0).img = scaledElem;

                exp -= 2;
            }
        }
        _gaussians = gaussians;
        return dogs; // TODO: by ref entgegen nehmen um copy zu vermeiden?
    }
}
