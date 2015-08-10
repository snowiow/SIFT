# SIFT Algorithm in C++
- [Intro](#intro)
- [Installation](#installation)
- [User Guide](#user-guide)
- [License](#license)

# Intro
This is a C++ implementation of the SIFT algorithm, which was originally presented by David G. Lowe
in the International Journal of Computer Vision 60 in January 2004. This algorithm is mostly implemented
after the principles described in Lowe's paper. Also some elements were taken from the lecture of Dr.
Mubarak Shah, which was held at the University of Central Florida.

# Installation
## Requirements
Vigra: A generic C++ library for image analysis(used for most of the calculations and image transformations)  
OpenCV: Open Source Computer Vision library(used for visualisation of the found sift features)  
Boost program_options: An easy to use layer for handling program arguments. Part of the Boost Library.  
CMake: A cross-platform open-source make system.  

## Installation Process
With the needed libraries installed and the help of CMake, the installation is as easy as three commands.
First download this repository by cloning or whatever way you prefer. Switch into the directory afterwards.
Now you want to make an extra directory for your build files. For Example:  
`mkdir build`  
Switch into the created directory  
`cd build`  
Then you want to create your prefered make files with cmake. On all unix like systems this will be 
GNU Makefiles.  
`cmake -G "Unix Makefiles" ..`  
For other supported build systems check the official documentation of cmake.  
The final step is to build your executable from the Makefiles  
`make`  
You should now have an executable named sift in your build directory. Please refer to the next section
to check how it is used and which possibilities you got, by executing it.

# User Guide
This is a CLI only application and there is no intention to make a GUI based application out of it. However the usage isn't very complicated. The easiest way to start of is just throwing an image in and get a new image back, with the sift features drawn on it. The file is called
`[file]_features.png`  
and can be found in the same directory as the original file. The command to produce this file is  
`./sift path/to/file.jpg`  
But there are many possible parameters on which you can screw the values. The following list shows the possibilities
```Options:
  --help                           Print help message
  -i [ --img ] arg                 The image on which sift will be executed
  -s [ --sigma ] arg (=1.60000002) The sigma value of the Gaussian calculations
  -k [ --k ] arg (=1.41421354)     The constant which is calculated on sigma 
                                   for the DoGs
  -o [ --octaves ] arg (=4)        How many octaves should be calculated
  -d [ --dogsPerEpoch ] arg (=3)   How many DoGs should be created per epoch
  -p [ --subpixel ] arg (=0)       Starts with the doubled size of initial
  ```
This overview can also be called by  
`./sift --help`  
The first flag is the shorthand and can generally be written by  
 `./sift path/to/file.jpg -[shorthand] [value]`  
 The second value inside the square brackets is the longhand and may be more readable, but it's also more typing.  The general theme looks like this  
 `./sift path/to/file.jpg --[longhand]=[value]`  
following after is the default value, which is set, if no argument is given by the user. These default values are based on the studies of David Lowe's paper and seem to give the most stable results overall.  The following chapters cover every flag in detail
## -i [ --img ] arg
The flag indicator is optional as seen in the first example of the user guide. This is also the only possible option which can be handled without a flag indicator. It takes an image file(tested: jpg, png) in RGB or Greyscale.  Currently the algorithm gets slower by the size of the image dramatically. A small list with different image sizes shows the calculation time on my PC(i7 vPro)
- ~300x300 px: ~0.7 seconds
- ~600x600 px: ~15 seconds
- ~1500x1500 px: ~11 minutes  
  
## -s [ --sigma ] arg (=1.60000002)
sigma is the standard deviation of the gaussian curve. It is used extensively throughout the algorithm. For example when creating the Difference of Gaussian(DoG) pyramid. 

## -k [ --k ] arg (=1.41421354)
k is the constant, which is calculated onto sigma in each step of the Gaussian creation process. For example the process in the first octave of the algorithm looks like the following:  
```image 1: sigma 
image 2: sigma * k ^ 1 = sigma * k  
image 3: sigma * k ^ 2 = ...  
```
  
