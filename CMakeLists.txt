cmake_minimum_required(VERSION 2.8.4) 
set(CMAKE_CXX_STANDARD 11)
project(unsharp_mask)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS} Cpp_common)
link_libraries(${OpenCL_LIBRARY})

add_executable(unsharp_mask unsharp_mask.cpp unsharp_mask.hpp add_weighted.hpp ppm.hpp util.hpp kernel.cl)

set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT unsharp_mask)
