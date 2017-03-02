project(timer)

include(GenerateExportHeader)

INCLUDE_DIRECTORIES(./)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

ADD_LIBRARY(timer timer.hpp timer.cpp)

generate_export_header(timer)
target_include_directories(timer PUBLIC ${CMAKE_CURRENT_BINARY_DIR})