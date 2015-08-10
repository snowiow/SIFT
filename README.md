# SIFT Algorithm in C++
- [Intro](#intro)
- [Installation](#installation)
- [User Guide](#user-guide)
- [License](#license)

# Intro
This is a C++ implementation of the SIFT algorithm, which was originally presented by David G. Lowe
in the International Journal of Computer Vision 60 in January 2004. This algorithm is mostly implemented
after the principles described in Lowe's paper. Also some elements were taken from the lecture of Dr.
Mubarak Shahi, whcih was held at the University of Central Florida.

# Installation
## Requirements
Vigra: A generic C++ library for image analysis(used for most of the calculations and image transformations)
OpenCV: Open Source Computer Vision library(used for visualisation of the found sift features)
Boost program_options: An easy to use layer for handling program arguments. Part of the Boost Library.
CMake: A cross-platform open-source make system.

## Installation Process
With the needed libraries installed and the help of CMake, the isntallation is as easy as three commands.
First download this repository by cloning or whatever way you prefer. Switch into the directory afterwards.
Now you want to make an extra directory for your build files. For Example:
`mkdir build
Switch into the created directory
`cd build
Then you want to create your prefered make files from cmake. On all unix like systems this will be 
GNU Makefiles.
`cmake -G "Unix Makefiles" ..
For other supported build systems check the official documentation of cmake.
The final step is to build your executable from the Makefiles
`make
You should now have an executable named sift in your build directory. Please refer to the next section
to check how it is used and which possibilities you got, by executing it.
