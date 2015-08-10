cmake_minimum_required(VERSION 3.2)
project(sift)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -O3")

FIND_PACKAGE(Vigra)
FIND_PACKAGE(OpenCV)
FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)

set(HEADER_FILES sift.hpp types.hpp point.hpp matrix.hpp algorithms.hpp octaveelem.hpp interestpoint.hpp)
set(SOURCE_FILES algorithms.cpp sift.cpp main.cpp  )
add_executable(sift ${SOURCE_FILES})
TARGET_INCLUDE_DIRECTORIES(sift PUBLIC ${Vigra_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
TARGET_LINK_LIBRARIES(sift vigraimpex ${OpenCV_LIBS} ${Boost_LIBRARIES})
