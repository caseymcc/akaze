/**
 * @file utils.h
 * @brief Some utilities functions
 * @date Oct 07, 2014
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#ifndef AKAZE_UTILS_H_
#define AKAZE_UTILS_H_

 //file generated during cmake build
#include "akaze_export.h"

/* ************************************************************************* */
#include "cimg/CImg.h"
#include "AKAZE.h"

// System
#include <vector>
#include <iostream>
#include <iomanip>

/* ************************************************************************* */
// Stringify common types such as int, double and others.
template <typename T> inline std::string to_string(const T& x) {
  std::stringstream oss;
  oss << x;
  return oss.str();
}

// Stringify and format integral types as follows:
// to_formatted_string(  1, 2) produces string:  '01'
// to_formatted_string(  5, 2) produces string:  '05'
// to_formatted_string( 19, 2) produces string:  '19'
// to_formatted_string( 19, 3) produces string: '019'
template <typename Integer>
inline std::string to_formatted_string(Integer x, int num_digits) {
  std::stringstream oss;
  oss << std::setfill('0') << std::setw(num_digits) << x;
  return oss.str();
}

/* ************************************************************************* */

// Converts to a CImg type and adjusts the scale for image displaying.
AKAZE_EXPORT void ConvertEigenToCImg(const RowMatrixXf& mat,
                        cimg_library::CImg<float>& cimg);

/* ************************************************************************* */

// Converts a CImg object to a grayscale floating point Eigen matrix.
AKAZE_EXPORT void ConvertCImgToEigen(const cimg_library::CImg<float>& image,
                        RowMatrixXf& eigen_image);

/* ************************************************************************* */
// Saves Eigen matrix as image
AKAZE_EXPORT void saveMatrixAsImage(const RowMatrixXf& mat, std::string fileName);

/* ************************************************************************* */
// Saves Eigen matrix as csv file
AKAZE_EXPORT void saveMatrixAsCsvFile(const RowMatrixXf& mat, std::string fileName);

/* ************************************************************************* */
// This function matches the descriptors from floating point AKAZE methods using
// L2 distance and the nearest neighbor distance ratio (Lowes ratio)
AKAZE_EXPORT void match_features(const libAKAZE::Descriptors &desc1,
    const libAKAZE::Descriptors &desc2,
    const double ratio,
    std::vector<std::pair<int, int> >& matches);


void match_features(const float *desc1, size_t desc1Count,
    const float *desc2, size_t desc2Count, size_t stride,
    const double ratio, std::vector<std::pair<int, int> >& matches);

// This function matches the descriptors from binary AKAZE (MLDB) methods using
// Hamming distance and the nearest neighbor distance ratio (Lowes ratio)
void match_binary_features(const uint8_t *desc1, size_t desc1Count,
    const uint8_t *desc2, size_t desc2Count, size_t stride,
    const double ratio, std::vector<std::pair<int, int> >& matches);

/// This function computes the set of inliers given a ground truth homography
/// @param kpts1 Keypoints from first image
/// @param kpts2 Keypoints from second image
/// @param matches Vector of putative matches
/// @param inliers Vector of inliers
/// @param H Ground truth homography matrix 3x3
/// @param min_error The minimum pixelic error to accept an inlier
AKAZE_EXPORT void compute_inliers_homography(
    const std::vector<libAKAZE::Keypoint>& kpts1,
    const std::vector<libAKAZE::Keypoint>& kpts2,
    const std::vector<std::pair<int, int> >& matches,
    std::vector<std::pair<int, int> >& inliers, const Eigen::Matrix3f& H,
    float min_error);

/// This function draws the list of detected keypoints
AKAZE_EXPORT void draw_keypoints(cimg_library::CImg<float>& img,
                    const std::vector<libAKAZE::Keypoint>& kpts);

/// This function draws the list of detected keypoints as vectors
AKAZE_EXPORT void draw_keypoints_vector(cimg_library::CImg<float>& image,
    const std::vector<libAKAZE::Keypoint>& kpts);

AKAZE_EXPORT void draw_matches(cimg_library::CImg<float>& image1,
                  cimg_library::CImg<float>& image2,
                  const std::vector<libAKAZE::Keypoint>& kpts1,
                  const std::vector<libAKAZE::Keypoint>& kpts2,
                  const std::vector<std::pair<int, int> >& matches,
                  cimg_library::CImg<float>& matched_image);

/// This function saves the interest points to a regular ASCII file
/// @note The format is compatible with Mikolajczyk and Schmid evaluation
/// @param outFile Name of the output file where the points will be stored
/// @param kpts Vector of points of interest
/// @param desc Matrix that contains the extracted descriptors
/// @param save_desc Set to 1 if we want to save the descriptors
AKAZE_EXPORT int save_keypoints(const std::string& outFile,
    const std::vector<libAKAZE::Keypoint>& kpts,
    //                   const std::vector<libAKAZE::Vector64f>& desc,
    const libAKAZE::Descriptors &desc,
    bool save_desc);

AKAZE_EXPORT int save_keypoints(const std::string& outFile,
    const std::vector<libAKAZE::Keypoint>& kpts,
//  const std::vector<libAKAZE::BinaryVectorX>& desc,
    const libAKAZE::Descriptors &desc,
    bool save_desc);

#ifdef AKAZE_USE_JSON
AKAZE_EXPORT int save_keypoints_json(const std::string& outFile,
    const std::vector<libAKAZE::Keypoint>& kpts,
//    const std::vector<libAKAZE::BinaryVectorX>& desc,
    const libAKAZE::Descriptors &desc,
    bool save_desc);
#endif //AKAZE_USE_JSON

/// Function for reading the ground truth homography from a txt file
AKAZE_EXPORT bool read_homography(const std::string& hFile, Eigen::Matrix3f& H1toN);

/// This function shows the possible command line configuration options
AKAZE_EXPORT void show_input_options_help(int example);

#endif  // AKAZE_UTILS_H_
