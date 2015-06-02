#include <iostream>

#include <vigra/impex.hxx>
#include <vigra/multi_array.hxx>

#include "sift.hpp"

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
        vigra::MultiArray<2, float> img(vigra::Shape2(info.shape()));
        vigra::importImage(info, img);

        Sift sift;
        sift.calculate(img, 3);
    } catch (std::exception& ex) {
        std::cerr << ex.what() << std::endl;
    }
    
    return 0;
}
