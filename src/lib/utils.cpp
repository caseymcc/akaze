//=============================================================================
//
// utils.cpp
// Authors: Pablo F. Alcantarilla (1), Jesus Nuevo (2)
// Institutions: Toshiba Research Europe Ltd (1)
//               TrueVision Solutions (2)
//
// Date: 07/10/2014
// Email: pablofdezalc@gmail.com
//
// AKAZE Features Copyright 2014, Pablo F. Alcantarilla, Jesus Nuevo
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file utils.cpp
 * @brief Some utilities functions
 * @date Oct 07, 2014
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#include "utils.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

// System
#include <algorithm>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#ifdef AKAZE_USE_JSON
#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/writer.h"
#include "rapidjson/prettywriter.h"
#endif //AKAZE_USE_JSON

/* ************************************************************************* */

// Converts to a CImg type and adjusts the scale for image displaying.
void ConvertEigenToCImg(const RowMatrixXf& mat,
                        cimg_library::CImg<float>& cimg) {
  cimg.resize(mat.cols(), mat.rows());
  const float min_coeff = mat.minCoeff();
  const float max_coeff = mat.maxCoeff();
  for (int y = 0; y < mat.rows(); y++) {
    for (int x = 0; x < mat.cols(); x++) {
      cimg(x, y) = (mat(y, x) - min_coeff) / (max_coeff - min_coeff);
    }
  }
}

/* ************************************************************************* */

void ConvertCImgToEigen(const cimg_library::CImg<float>& image,
                        RowMatrixXf& eigen_image) {
  cimg_library::CImg<float> grayscale_image;

  // Convert to grayscale if needed.
  if (image.spectrum() == 1) {
    grayscale_image = image;
  } else {
    grayscale_image = image.get_RGBtoYCbCr().channel(0);
  }
  // Copy the grayscale image to the eigen matrix.
  eigen_image = Eigen::Map<const Eigen::Matrix<
      float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(
      grayscale_image.data(),
      grayscale_image.height(),
      grayscale_image.width());
}

/* ************************************************************************* */
void saveMatrixAsImage(const RowMatrixXf& mat, std::string fileName)
{
    cimg_library::CImg<float> image;
    ConvertEigenToCImg(mat, image);
    std::string outputFile;

    image.normalize(0, 255);
    image.save(fileName.c_str());
}

/* ************************************************************************* */
void saveMatrixAsCsvFile(const RowMatrixXf& mat, std::string fileName)
{
	std::ofstream file(fileName);
	float value;

	file<<std::setprecision(4);
	for(int rows=0; rows<mat.rows(); rows++)
	{
		file<<mat(rows, 0);
		for(int cols=1; cols<mat.cols(); cols++)
		{
			file<<", "<<mat(rows, cols);
		}
		file<<"\n";
	}
}

/* ************************************************************************* */

// Build a simple lookup table for the hamming distance of xor-ing two 8-bit
// numbers.
//int HammingDistance(const libAKAZE::BinaryVectorX& desc1,
//                    const libAKAZE::BinaryVectorX& desc2)
int HammingDistance(const unsigned char *char1, const unsigned char *char2,
    size_t num_bytes)
{
  static const unsigned char pop_count_table[] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2,
    3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3,
    3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3,
    4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4,
    3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5,
    6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4,
    4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5,
    6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5,
    3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3,
    4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6,
    6, 7, 6, 7, 7, 8
  };

  int result = 0;
//  const unsigned char* char1 = desc1.data();
//  const unsigned char* char2 = desc2.data();
//  const int num_bytes = desc1.size() / (8 * sizeof(unsigned char));
  for (size_t i = 0; i < num_bytes; i++) {
    result += pop_count_table[char1[i] ^ char2[i]];
  }

  return result;
}

void match_features(const libAKAZE::Descriptors &desc1,
    const libAKAZE::Descriptors &desc2,
    const double ratio,
    std::vector<std::pair<int, int> >& matches)
{
    if(desc1.isBinary())
    {
        if(desc2.isBinary())
        {
            match_binary_features(desc1.binaryData(), desc1.size(), desc2.binaryData(), desc2.size(), desc1.descriptorSize(), ratio, matches);
            return;
        }
    }
    else
    {
        if(!desc2.isBinary())
        {
            match_features(desc1.floatData(), desc1.size(), desc2.floatData(), desc2.size(), desc1.descriptorSize(), ratio, matches);
            return;
        }
    }
}

void match_features(const float *desc1, size_t desc1Count,
    const float *desc2, size_t desc2Count, size_t stride,
    const double ratio, std::vector<std::pair<int, int> >& matches)
{
  // Find all desc1 -> desc2 matches.
  typedef std::pair<float, int> MatchDistance;

  std::vector<std::vector<MatchDistance> > match_distances(desc1Count);
  #ifdef AKAZE_USE_OPENMP
  #pragma omp parallel for
  #endif

  for (int i = 0; i < desc1Count; i++)
  {
      Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 1> > descMat1(&desc1[i*stride], desc1Count);

    match_distances[i].resize(desc2Count);
    for (int j = 0; j < desc2Count; j++)
    {
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 1> > descMat2(&desc2[j*stride], desc2Count);

//      const float distance = (desc1[i] - desc2[j]).squaredNorm();
        const float distance=(descMat1-descMat2).squaredNorm();
      match_distances[i][j] = std::make_pair(distance, j);
    }
  }

  // Only save the matches that pass the lowes ratio test.
  matches.reserve(desc1Count);
  
  for (int i = 0; i < match_distances.size(); i++)
  {
    // Get the top 2 matches.
    std::partial_sort(match_distances[i].begin(),
                      match_distances[i].begin() + 2,
                      match_distances[i].end());
    if (match_distances[i][0].first / match_distances[i][1].first < ratio) {
      matches.push_back(std::make_pair(i, match_distances[i][0].second));
    }
  }
}

void match_binary_features(const uint8_t *desc1, size_t desc1Count,
                    const uint8_t *desc2, size_t desc2Count, size_t stride,
                    const double ratio, std::vector<std::pair<int, int> >& matches)
{
  // Find all desc1 -> desc2 matches.
  typedef std::pair<int, int> MatchDistance;

  std::vector<std::vector<MatchDistance> > match_distances(desc1Count);
#ifdef AKAZE_USE_OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < desc1Count; i++)
  {
    match_distances[i].resize(desc2Count);
    for (int j = 0; j < desc2Count; j++)
    {
        const int distance=HammingDistance(&desc1[i*stride], &desc2[j*stride], stride);
      match_distances[i][j] = std::make_pair(distance, j);
    }
  }

  // Only save the matches that pass the lowes ratio test.
  matches.reserve(desc1Count);
  for (int i = 0; i < match_distances.size(); i++) {
    // Get the top 2 matches.
    std::partial_sort(match_distances[i].begin(),
                      match_distances[i].begin() + 2,
                      match_distances[i].end());
    if (static_cast<double>(match_distances[i][0].first) /
            match_distances[i][1].first <
        ratio) {
      matches.push_back(std::make_pair(i, match_distances[i][0].second));
    }
  }
}

void compute_inliers_homography(
    const std::vector<libAKAZE::Keypoint>& kpts1,
    const std::vector<libAKAZE::Keypoint>& kpts2,
    const std::vector<std::pair<int, int> >& matches,
    std::vector<std::pair<int, int> >& inliers,
    const Eigen::Matrix3f& H,
    float min_error) {
  inliers.reserve(matches.size());
  for (int i = 0; i < matches.size(); i++) {
    const Eigen::Vector3f& kpt1 = kpts1[matches[i].first].pt.homogeneous();
    const Eigen::Vector3f& kpt2 = kpts2[matches[i].second].pt.homogeneous();
    const Eigen::Vector3f projection = H * kpt1;
    if ((projection - kpt2).norm() < min_error) {
      inliers.push_back(matches[i]);
    }
  }
}

/* ************************************************************************* */
void draw_keypoints(cimg_library::CImg<float>& image,
                    const std::vector<libAKAZE::Keypoint>& kpts) {
  int x = 0, y = 0;
  float radius = 0.0;

  for (size_t i = 0; i < kpts.size(); i++) {
    x = (int)(kpts[i].pt.x() + .5);
    y = (int)(kpts[i].pt.y() + .5);
    radius = kpts[i].size / 2.0;
    float color1[3] = {0, 255, 0};
    float color2[3] = {0, 0, 255};
    image.draw_circle(x, y, radius * 2.5, color1, 1, 0L);
    image.draw_circle(x, y, 1.0, color2);
  }
}

void draw_keypoints_vector(cimg_library::CImg<float>& image,
    const std::vector<libAKAZE::Keypoint>& kpts)
{
    int x=0, y=0;
    float radius=0.0;

    for(size_t i=0; i < kpts.size(); i++)
    {
        x=(int)(kpts[i].pt.x()+.5);
        y=(int)(kpts[i].pt.y()+.5);
        radius=kpts[i].size/2.0;
        float green[3]={0, 255, 0};
        float blue[3]={0, 0, 255};
        float red[3]={255, 0, 0};
        float yellow[3]={255, 255, 0};
        float purple[3]={255, 0, 255};

        float angle=kpts[i].angle;

        int endX=2.5*radius*cos(angle)+x;
        int endY=2.5*radius*sin(angle)+y;
        
        image.draw_arrow(x, y, endX, endY, green);
#ifdef TRACK_REMOVED
        if(kpts[i].removed == 0)
            image.draw_circle(x, y, 1.0, blue);
        else if(kpts[i].removed==1)
            image.draw_circle(x, y, 1.0, red);
        else if(kpts[i].removed==2)
            image.draw_circle(x, y, 1.0, yellow);
        else if(kpts[i].removed==3)
            image.draw_circle(x, y, 1.0, purple);
#else //TRACK_REMOVED
        image.draw_circle(x, y, 1.0, blue);
#endif //TRACK_REMOVED
    }
}

void draw_matches(cimg_library::CImg<float>& image1,
                  cimg_library::CImg<float>& image2,
                  const std::vector<libAKAZE::Keypoint>& kpts1,
                  const std::vector<libAKAZE::Keypoint>& kpts2,
                  const std::vector<std::pair<int, int> >& matches,
                  cimg_library::CImg<float>& matched_image) {
  matched_image = image1;
  matched_image.append(image2);
  for (int i = 0; i < matches.size(); i++) {
    float color[3] = {255, 0, 0};
    const Eigen::Vector2f& pt1 = kpts1[matches[i].first].pt;
    const Eigen::Vector2f& pt2 = kpts2[matches[i].second].pt;
    matched_image.draw_line(pt1.x(),
                            pt1.y(),
                            pt2.x() + image1.width(),
                            pt2.y(),
                            color);
  }
}

/* ************************************************************************* */
int save_keypoints(const std::string& outFile,
                   const std::vector<libAKAZE::Keypoint>& kpts,
                   const std::vector<libAKAZE::Vector64f>& desc,
                   bool save_desc) {
  if (kpts.size() == 0) {
    std::cerr << "No keypoints exist." << std::endl;
    return -1;
  }

  int nkpts = 0, dsize = 0;
  float sc = 0.0;

  nkpts = (int)(kpts.size());
  dsize = (int)(desc[0].size());

  std::ofstream ipfile(outFile.c_str());

  if (!ipfile) {
    std::cerr << "Couldn't open file '" << outFile << "'!" << std::endl;
    return -1;
  }

  if (!save_desc) {
    ipfile << 1 << std::endl << nkpts << std::endl;
  } else {
    ipfile << dsize << std::endl << nkpts << std::endl;
  }

  // Save interest point with descriptor in the format of Krystian Mikolajczyk
  // for reasons of comparison with other descriptors
  for (int i = 0; i < nkpts; i++) {
    // Radius of the keypoint
    sc = (kpts[i].size);
    sc *= sc;

    ipfile << kpts[i].pt.x()        /* x-location of the interest point */
           << " " << kpts[i].pt.y() /* y-location of the interest point */
           << " " << 1.0 / sc     /* 1/r^2 */
           << " " << 0.0 << " " << 1.0 / sc; /* 1/r^2 */

    // Here comes the descriptor
    for (int j = 0; j < dsize; j++) {
      ipfile << " " << desc[i](j);
    }

    ipfile << std::endl;
  }

  // Close the txt file
  ipfile.close();

  return 0;
}

int save_keypoints(const std::string& outFile,
                   const std::vector<libAKAZE::Keypoint>& kpts,
                    const libAKAZE::Descriptors &desc,
                   bool save_desc) {
  if (kpts.size() == 0) {
    std::cerr << "No keypoints exist." << std::endl;
    return -1;
  }

  int nkpts = 0, dsize = 0;
  float sc = 0.0;

  nkpts = (int)(kpts.size());
  dsize=desc.descriptorSize();
  const uint8_t *data=desc.binaryData();

  std::ofstream ipfile(outFile.c_str());

  if (!ipfile) {
    std::cerr << "Couldn't open file '" << outFile << "'!" << std::endl;
    return -1;
  }

  if (!save_desc) {
    ipfile << 1 << std::endl << nkpts << std::endl;
  } else {
    ipfile << dsize << std::endl << nkpts << std::endl;
  }

  // Save interest point with descriptor in the format of Krystian Mikolajczyk
  // for reasons of comparison with other descriptors
  for (int i = 0; i < nkpts; i++) {
    // Radius of the keypoint
    sc = (kpts[i].size);
    sc *= sc;

    ipfile << kpts[i].pt.x()        /* x-location of the interest point */
           << " " << kpts[i].pt.y() /* y-location of the interest point */
           << " " << 1.0 / sc     /* 1/r^2 */
           << " " << 0.0 << " " << 1.0 / sc; /* 1/r^2 */

    // Here comes the descriptor
    size_t pos=i*dsize;
    for(int j=0; j < dsize; j++)
    {
      ipfile << " " << data[pos+j];
    }

    ipfile << std::endl;
  }

  // Close the txt file
  ipfile.close();

  return 0;
}

#ifdef AKAZE_USE_JSON
int save_keypoints_json(const std::string &outFile,
    const std::vector<libAKAZE::Keypoint> &keypoints,
    const libAKAZE::Descriptors &descriptors,
    bool save_desc)
{
    if(keypoints.size()==0)
    {
        std::cerr<<"No keypoints exist."<<std::endl;
        return -1;
    }

    FILE *jsonFile=fopen(outFile.c_str(), "wb");

    if(jsonFile==NULL)
        return -1;

    std::vector<char> buffer(65536);
    rapidjson::FileWriteStream *fileStream=new rapidjson::FileWriteStream(jsonFile, buffer.data(), buffer.size());
    rapidjson::PrettyWriter<rapidjson::FileWriteStream> *writer=new rapidjson::PrettyWriter<rapidjson::FileWriteStream>(*fileStream);

    writer->StartArray();
    for(size_t i=0; i<keypoints.size(); ++i)
    {
        const libAKAZE::Keypoint &keypoint=keypoints[i];

        writer->StartObject();
        
        writer->Key("class_id");
        writer->Int(keypoint.class_id);
        writer->Key("octave");
        writer->Int(keypoint.octave);

        writer->Key("ptX");
        writer->Double(keypoint.pt.x());
        writer->Key("ptX");
        writer->Double(keypoint.pt.y());
        writer->Key("size");
        writer->Double(keypoint.size);
        writer->Key("angle");
        writer->Double(keypoint.angle);
        writer->Key("response");
        writer->Double(keypoint.response);
#ifdef TRACK_REMOVED
        writer->Key("removed");
        writer->Double(keypoint.removed);
#endif //TRACK_REMOVED

        if(save_desc)
        {
            const uint8_t *data=descriptors.binaryData();
            size_t descriptorSize=descriptors.descriptorSize();

            writer->Key("descriptors");
            writer->StartArray();

            for(size_t j=0; j<descriptorSize; ++j)
            {
                writer->Int(data[j]);
            }
            writer->EndArray();
        }

        writer->EndObject();
    }
    writer->EndArray();

    fclose(jsonFile);
    return 0;
}
#endif //AKAZE_USE_JSON

/* ************************************************************************* */
bool read_homography(const std::string& hFile, Eigen::Matrix3f& H1toN) {

  float h11 = 0.0, h12 = 0.0, h13 = 0.0;
  float h21 = 0.0, h22 = 0.0, h23 = 0.0;
  float h31 = 0.0, h32 = 0.0, h33 = 0.0;
  const int tmp_buf_size = 256;
  char tmp_buf[tmp_buf_size];

  // Allocate memory for the Homography matrices
  H1toN.setZero();

  std::string filename(hFile);
  std::ifstream pf;
  pf.open(filename.c_str(), std::ifstream::in);

  if (!pf.is_open()) return false;

  pf.getline(tmp_buf, tmp_buf_size);
  sscanf(tmp_buf, "%f %f %f", &h11, &h12, &h13);

  pf.getline(tmp_buf, tmp_buf_size);
  sscanf(tmp_buf, "%f %f %f", &h21, &h22, &h23);

  pf.getline(tmp_buf, tmp_buf_size);
  sscanf(tmp_buf, "%f %f %f", &h31, &h32, &h33);

  pf.close();

  H1toN(0, 0) = h11 / h33;
  H1toN(0, 1) = h12 / h33;
  H1toN(0, 2) = h13 / h33;

  H1toN(1, 0) = h21 / h33;
  H1toN(1, 1) = h22 / h33;
  H1toN(1, 2) = h23 / h33;

  H1toN(2, 0) = h31 / h33;
  H1toN(2, 1) = h32 / h33;
  H1toN(2, 2) = h33 / h33;

  return true;
}

/* ************************************************************************* */
const size_t length = std::string("--descriptor_channels").size() + 2;
static inline std::ostream& cout_help() {
  std::cout << std::setw(length);
  return std::cout;
}

/* ************************************************************************* */
void show_input_options_help(int example) {

  fflush(stdout);
  std::cout << "A-KAZE Features" << std::endl;
  std::cout << "Usage: ";

  if (example == 0) {
    std::cout << "./akaze_features img.jpg output.jpg [options]" << std::endl;
  } else if (example == 1) {
    std::cout << "./akaze_match img1.jpg img2.pgm homography.txt output.jpg [options]"
         << std::endl;
  }

  std::cout << std::endl;
  cout_help() << "homography.txt is optional for image matching. If it is "
                 "present, only the inliers are shown." << std::endl;

  std::cout << std::endl;
  cout_help() << "Options below are not mandatory. Unless specified, default "
                 "arguments are used." << std::endl << std::endl;

  // Justify on the left
  std::cout << std::left;

  // Generalities
  cout_help() << "--help"
              << "Show the command line options" << std::endl;
  cout_help() << "--verbose "
              << "Verbosity is required" << std::endl;
  cout_help() << "--num_threads"
              << "Number of threads to run AKAZE with" << std::endl;
  cout_help() << std::endl;

  // Scale-space parameters
  cout_help() << "--soffset"
              << "Base scale offset (sigma units)" << std::endl;
  cout_help() << "--omax"
              << "Maximum octave of image evolution" << std::endl;
  cout_help() << "--nsublevels"
              << "Number of sublevels per octave" << std::endl;
  cout_help() << "--diffusivity"
              << "Diffusivity function. Possible values:" << std::endl;
  cout_help() << " "
              << "0 -> Perona-Malik, g1 = exp(-|dL|^2/k^2)" << std::endl;
  cout_help() << " "
              << "1 -> Perona-Malik, g2 = 1 / (1 + dL^2 / k^2)" << std::endl;
  cout_help() << " "
              << "2 -> Weickert diffusivity" << std::endl;
  cout_help() << " "
              << "3 -> Charbonnier diffusivity" << std::endl;
  cout_help() << std::endl;

  // Feature detection parameters.
  cout_help() << "--dthreshold"
              << "Feature detector threshold response for keypoints" << std::endl;
  cout_help() << " "
              << "(0.001 can be a good value)" << std::endl;
  cout_help() << std::endl;
  cout_help() << std::endl;

  // Descriptor parameters.
  cout_help() << "--descriptor"
              << "Descriptor Type. Possible values:" << std::endl;
  cout_help() << " "
              << "0 -> SURF_UPRIGHT" << std::endl;
  cout_help() << " "
              << "1 -> SURF" << std::endl;
  cout_help() << " "
              << "2 -> M-SURF_UPRIGHT," << std::endl;
  cout_help() << " "
              << "3 -> M-SURF" << std::endl;
  cout_help() << " "
              << "4 -> M-LDB_UPRIGHT" << std::endl;
  cout_help() << " "
              << "5 -> M-LDB" << std::endl;
  cout_help() << std::endl;

  cout_help() << "--descriptor_channels "
              << "Descriptor Channels for M-LDB. Valid values: " << std::endl;
  cout_help() << " "
              << "1 -> intensity" << std::endl;
  cout_help() << " "
              << "2 -> intensity + gradient magnitude" << std::endl;
  cout_help() << " "
              << "3 -> intensity + X and Y gradients" << std::endl;
  cout_help() << std::endl;

  cout_help() << "--descriptor_size"
              << "Descriptor size for M-LDB in bits." << std::endl;
  cout_help() << " "
              << "0: means the full length descriptor (486)!!" << std::endl;
  cout_help() << std::endl;

  // Save results?
  cout_help() << "--show_results"
              << "Possible values below:" << std::endl;
  cout_help() << " "
              << "1 -> show detection results." << std::endl;
  cout_help() << " "
              << "0 -> don't show detection results" << std::endl;
  cout_help() << std::endl;
}
