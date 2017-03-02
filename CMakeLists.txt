include(${CMAKE_CURRENT_LIST_DIR}/cmake/Hunter/HunterGate.cmake)

HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.17.12.tar.gz"
    SHA1 "74cfa490ef0d3ff8916523bbf5add53049aad7be"
    LOCAL
)

project(AKAZE)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/CMakeModules)

cmake_minimum_required (VERSION 3.0)
if (COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif ()

# ============================================================================ #
# Build Type. By default is release
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug, Release" FORCE)

# ============================================================================ #
# Where to output binaries and libraries
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")

# ============================================================================ #
# Where to install the AKAZE library
set(AKAZE_INSTALL_PREFIX "/usr/local/lib" CACHE PATH "Installation Directory")
set(AKAZE_INCLUDE_PREFIX "/usr/local/include" CACHE PATH "Includes Directory")

# ============================================================================ #
# CPP sources
message(STATUS ">>> Adding src subdirectory")
add_subdirectory(src)
