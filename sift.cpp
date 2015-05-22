#include "sift.hpp"

using namespace vigra;
using namespace vigra::multi_math;

void Sift::calculate(vigra::MultiArray<2, float> img, float sigma, float k) {
    auto first = convolveWithGauss(img, sigma);
    std::vector<MultiArray<2, float>> laplacians;
    std::vector<MultiArray<2, float>> gaussians;
    gaussians.push_back(first);

    //first epoch
    unsigned int laplacionsPerEpoch = 4;

    for (unsigned int i = 1; i < laplacionsPerEpoch + 1; i++) {
        MultiArray<2, float> gaussian = convolveWithGauss(img, std::pow(k, i) *
            sigma);

        gaussians.push_back(gaussian);
        MultiArray<2, float> laplacian = laplacianOfGaussian(gaussians[i - 1],
             gaussians[i]);

        laplacians.push_back(laplacian);
    }

    //Save laplacianOfGaussian for viewing
    for (unsigned int i = 0; i < laplacians.size(); i++) {
        std::string fnStr = "laplacian" + std::to_string(i) + ".png";
        char fn[fnStr.size()];
        strcpy(fn, fnStr.c_str());
        exportImage(laplacians[i], ImageExportInfo(fnStr.c_str()));
    }
}

MultiArray<2, float> Sift::reduceToNextLevel(const MultiArray<2, float> & in)
{
    // image size at current level
    int height = in.height();
    int width = in.width();

    // image size at next smaller level
    int newheight = (height + 1) / 2;
    int newwidth = (width + 1) / 2;

    // resize result image to appropriate size
    MultiArray<2, float> out(Shape2(newwidth, newheight));

    // define a Gaussian kernel (size 5x1)
    Kernel1D<double> filter;
    //Inits the filter with the 5 values from -2 the value to +2 the value
    filter.initExplicitly(-2, 2) = 0.05, 0.25, 0.4, 0.25, 0.05;

    MultiArray<2, float> tmpimage1(Shape2(width, height));
    MultiArray<2, float> tmpimage2(Shape2(width, height));

    //convolves every row from in to out with the filter
    separableConvolveX(in, tmpimage1, filter);
    //same for y
    separableConvolveY(tmpimage1, tmpimage2, filter);

    // downsample smoothed image
    resizeImageNoInterpolation(tmpimage2, out);
    return out;
}

MultiArray<2, float> Sift::convolveWithGauss(const MultiArray<2, float>& input, float sigma) {
    Kernel1D<double> filter;
    filter.initGaussian(sigma);

    MultiArray<2, float> tmp(input.shape());
    MultiArray<2, float> result(input.shape());
    separableConvolveX(input, tmp, filter);
    separableConvolveY(tmp, result, filter);
    return result;
}

MultiArray<2, float> Sift::laplacianOfGaussian(
    const MultiArray<2, float>& lower, const MultiArray<2, float> higher) {

   MultiArray<2, float>result = higher - lower;
   return result;
}
