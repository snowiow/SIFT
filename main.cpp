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
        vigra::MultiArray<2, f32_t> img(vigra::Shape2(info.shape()));
        vigra::importImage(info, img);

        sift::Sift sift(3, 4);
        sift.calculate(img);
    } catch (std::exception& ex) {
        std::cerr << ex.what() << std::endl;
    }
    
    return 0;
}
