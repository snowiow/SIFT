cmake_minimum_required(VERSION 3.2)
project(sift)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

FIND_PACKAGE(Vigra)

set(HEADER_FILES sift.hpp)
set(SOURCE_FILES main.cpp sift.cpp)

add_executable(sift ${SOURCE_FILES})
TARGET_INCLUDE_DIRECTORIES(sift PUBLIC ${Vigra_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(sift vigraimpex)
