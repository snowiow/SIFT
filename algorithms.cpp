#include "algorithms.hpp"

#include <vigra/convolution.hxx>
#include <vigra/linear_algebra.hxx>

using namespace vigra::linalg;

namespace sift {
    namespace alg {
        const vigra::MultiArray<2, f32_t> convolveWithGauss(const vigra::MultiArray<2, f32_t>& input, 
                f32_t sigma) {

            vigra::Kernel1D<f32_t> filter;
            filter.initGaussian(sigma);
            vigra::MultiArray<2, f32_t> tmp(input.shape());
            vigra::MultiArray<2, f32_t> result(input.shape());

            separableConvolveX(input, tmp, filter);
            separableConvolveY(tmp, result, filter);

            return result;
        }

        const vigra::MultiArray<2, f32_t> reduceToNextLevel(const vigra::MultiArray<2, f32_t>& in, 
                f32_t sigma) {

            // image size at current level
            const vigra::Shape2 s((in.width()+ 1) / 2, (in.height() + 1) / 2);

            // resize result image to appropriate size
            vigra::MultiArray<2, f32_t> out(s);
            // downsample smoothed image
            resizeImageNoInterpolation(convolveWithGauss(in, sigma), out);

            return out; // TODO by ref entgegen nehmen um copy zu vermeiden?
        }

        const vigra::MultiArray<2, f32_t> dog(const vigra::MultiArray<2, f32_t>& lower, 
                const vigra::MultiArray<2, f32_t>& higher) {

            vigra::MultiArray<2, f32_t> result(vigra::Shape2(lower.shape()));
            for (u16_t x = 0; x < lower.shape(0); x++) {
                for (u16_t y = 0; y < lower.shape(1); y++) {
                    f32_t dif = higher(x, y) - lower(x, y);
                    result(x, y) = 128 + dif;
                }
            }
            return result;
        }

        const vigra::Matrix<f32_t> foDerivative(const vigra::MultiArray<2, f32_t> img[3], 
                const Point<u16_t, u16_t>& p) {

            f32_t dx = (img[1](p.x - 1, p.y) - img[1](p.x + 1, p.y)) / 2;
            f32_t dy = (img[1](p.x, p.y - 1) - img[1](p.x, p.y + 1)) / 2;
            f32_t ds = (img[0](p.x, p.y) - img[2](p.x, p.y)) / 2;
            vigra::Matrix<f32_t> result(vigra::Shape2(3, 1));
            result(0, 0) = dx;
            result(1, 0) = dy;
            result(2, 0) = ds;
            return result;
        }

        const vigra::Matrix<f32_t> soDerivative(const vigra::MultiArray<2, f32_t> img[3], 
                const Point<u16_t, u16_t>& p) {

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

        f32_t gradientMagnitude(const vigra::MultiArray<2, f32_t>& img, const Point<u16_t, u16_t>& p) {
            return std::sqrt(std::pow(img(p.x + 1, p.y) - img(p.x - 1, p.y), 2) + 
                    std::pow(img(p.x, p.y + 1) - img(p.x, p.y - 1), 2));
        }

        f32_t gradientOrientation(const vigra::MultiArray<2, f32_t>& img, const Point<u16_t, u16_t>& p) {
            const f32_t result = std::atan2(img(p.x, p.y + 1) - img(p.x, p.y - 1), img(p.x + 1, p.y) - img(p.x - 1, p.y));
            return std::fmod(result + 360, 360);
        }

        const std::array<f32_t, 36> orientationHistogram(
                const vigra::MultiArray<2, f32_t>& orientations,
                const vigra::MultiArray<2, f32_t>& magnitudes, 
                const vigra::MultiArray<2, f32_t>& current_gauss) {

            std::array<f32_t, 36> bins = {{0}};
            for (u16_t x = 0; x < orientations.width(); x++) {
                for (u16_t y = 0; y < orientations.height(); y++) {
                    f32_t sum = magnitudes(x, y) * current_gauss(x, y);
                    u16_t i = std::floor(orientations(x, y) / 10);
                    i = i > 35 ? 0 : i;
                    bins[i] += sum;
                }
            }
            return bins;
        }

        f32_t vertexParabola(const Point<u16_t, f32_t>& ln, const Point<u16_t, f32_t>& peak, 
                const Point<u16_t, f32_t>& rn) {

            vigra::MultiArray<2, f32_t> a(vigra::Shape2(3, 3));
            a(0, 0) = std::pow(ln.x, 2);
            a(1, 0) = std::pow(peak.x, 2);
            a(2, 0) = std::pow(rn.x, 2); 

            a(0, 1) = ln.x;
            a(1, 1) = peak.x;
            a(2, 1) = rn.x;

            a(0, 2) = 0;
            a(1, 2) = 0;
            a(2, 2) = 0;

            vigra::MultiArray<2, f32_t> b(vigra::Shape2(3, 1));
            b(0, 0) = ln.y;
            b(1, 0) = peak.y;
            b(2, 0) = rn.y;

            vigra::MultiArray<2, f32_t> res(vigra::Shape2(3, 1));
            linearSolve(a, b, res);

            return -res(1, 0) / (2 * res(0, 0));
        }
    }
}
