#include <vigra/impex.hxx>
#include <vigra/multi_array.hxx>

#include "sift.hpp"

/*
* Main Function takes a greyvalue image as input
*/
int main(int argc, char** argv) {
    vigra::ImageImportInfo info(argv[1]);
    vigra::MultiArray<2, float> img(vigra::Shape2(info.shape()));
    vigra::importImage(info, img);

    Sift sift;
    sift.calculate(img);
    
    return 0;
}
