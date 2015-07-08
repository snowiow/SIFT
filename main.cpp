#include <iostream>

#include <vector>
#include <string>

#include <vigra/impex.hxx>
#include <vigra/multi_array.hxx>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "sift.hpp"
#include "interestpoint.hpp"

/*
* Main Function takes a greyvalue image as input
*/
int main(int argc, char** argv) {
    std::string img_file;
    if (argc > 1)
        img_file = argv[1];
    else
        img_file = "lena.jpg";
    
    try {
        vigra::ImageImportInfo info(img_file.c_str());
        vigra::MultiArray<2, f32_t> img(vigra::Shape2(info.shape()));
        vigra::importImage(info, img);

        sift::Sift sift(3, 4);
        std::vector<sift::InterestPoint> interestPoints = sift.calculate(img);

        auto image = cv::imread(img_file.c_str(), CV_LOAD_IMAGE_COLOR);
        for (const sift::InterestPoint& p : interestPoints) {
            u16_t x = p.loc.x * std::pow(2, p.octave);
            u16_t y = p.loc.y * std::pow(2, p.octave);
            cv::RotatedRect r(cv::Point2f(x, y), 
                              cv::Size(p.scale * 10, p.scale * 10),
                              *(p.orientation.begin()));
            cv::Point2f points[4]; 
            r.points( points );
            cv::line(image, points[0], points[1], cv::Scalar(255, 0, 0));
            cv::line(image, points[0], points[3], cv::Scalar(255, 0, 0));
            cv::line(image, points[2], points[3], cv::Scalar(255, 0, 0));
            cv::line(image, points[1], points[2], cv::Scalar(255, 0, 0));
        }

        cv::imwrite(img_file + "_orientation.png", image);

    } catch (std::exception& ex) {
        std::cerr << ex.what() << std::endl;
    }
    
    return 0;
}
