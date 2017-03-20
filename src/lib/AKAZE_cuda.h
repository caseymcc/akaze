/**
 * @file AKAZE.h
 * @brief Main class for detecting and computing binary descriptors in an
 * accelerated nonlinear scale space
 * @date Oct 07, 2014
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#pragma once

/* ************************************************************************* */
#include "AKAZE.h"

#ifdef AKAZE_USE_CUDA
#include "cudaImage.h"
#include "cuda_akaze.h"

#ifdef USE_PYTHON
// Boost
#include <boost/python.hpp>
#endif

/* ************************************************************************* */
namespace libAKAZE
{
namespace cuda
{
class AKAZE_EXPORT Matcher
{

private:
    int maxnquery;
    unsigned char* descq_d;

    int maxntrain;
    unsigned char* desct_d;

    Match* dmatches_d;
    Match* dmatches_h;

    size_t pitch;

public:
    Matcher(): maxnquery(0), descq_d(NULL), maxntrain(0), desct_d(NULL),
        dmatches_d(0), dmatches_h(0), pitch(0) {}

    ~Matcher();

    // python
//        cv::Mat bfmatch_(cv::Mat desc_query, cv::Mat desc_train);

    void bfmatch(Descriptors &desc_query, Descriptors &desc_train,
        std::vector<std::vector<Match> > &dmatches);

};

class AKAZE_EXPORT AKAZE
{

  private:

    Options options_;                      ///< Configuration options for AKAZE
    std::vector<TEvolution> evolution_;         ///< Vector of nonlinear diffusion evolution

    /// FED parameters
    int ncycles_;                               ///< Number of cycles
    bool reordering_;                           ///< Flag for reordering time steps
    std::vector<std::vector<float > > tsteps_;  ///< Vector of FED dynamic time steps
    std::vector<int> nsteps_;                   ///< Vector of number of steps per cycle

    /// Matrices for the M-LDB descriptor computation
    Eigen::MatrixXi descriptorSamples_;
    Eigen::MatrixXi descriptorBits_;

    /// Computation times variables in ms
    Timing timing_;

    /// CUDA memory buffers
    float *cuda_memory;
    Keypoint *cuda_points;
    Keypoint *cuda_bufferpoints;
    Descriptors cuda_desc;

    float* cuda_descbuffer;
    int* cuda_ptindices;
    CudaImage *cuda_images;
    std::vector<CudaImage> cuda_buffers;
    int nump;

    int ncudaimages;                ///< Number of CUDA images allocated per octave
    int maxkeypoints;               ///< Maximum number of keypoints allocated

  public:

    /// AKAZE constructor with input options
    /// @param options AKAZE configuration options
    /// @note This constructor allocates memory for the nonlinear scale space
    AKAZE(const Options& options);

    /// Destructor
    ~AKAZE();

    /// Allocate the memory for the nonlinear scale space
    void Allocate_Memory_Evolution();

    /// This method creates the nonlinear scale space for a given image
    /// @param img Input image for which the nonlinear scale space needs to be created
    /// @return 0 if the nonlinear scale space was created successfully, -1 otherwise
    int Create_Nonlinear_Scale_Space(const RowMatrixXf& img);

    /// @brief This method selects interesting keypoints through the nonlinear scale space
    /// @param kpts Vector of detected keypoints
    void Feature_Detection(std::vector<Keypoint>& kpts);

    /// This method computes the feature detector response for the nonlinear scale space
    /// @note We use the Hessian determinant as the feature detector response
    void Compute_Determinant_Hessian_Response();

    /// This method computes the multiscale derivatives for the nonlinear scale space
    void Compute_Multiscale_Derivatives();

    /// This method finds extrema in the nonlinear scale space
    void Find_Scale_Space_Extrema(std::vector<Keypoint>& kpts);

    /// This method performs subpixel refinement of the detected keypoints fitting a quadratic
    void Do_Subpixel_Refinement(std::vector<Keypoint>& kpts);

    /// Feature description methods
#ifdef USE_PYTHON
    boost::python::tuple Compute_Descriptors_();
#endif // USE_PYTHON
    void Compute_Descriptors(std::vector<Keypoint> &kpts,
        Descriptors &desc);

    /// This method saves the scale space into jpg images
    void Save_Scale_Space();

    /// This method saves the feature detector responses of the nonlinear scale space into jpg images
    void Save_Detector_Responses();

    /// Display timing information
    void Show_Computation_Times() const;

    /// Return the computation times
    Timing Get_Computation_Times() const {
      return timing_;
    }
  };

  /* ************************************************************************* */

  /// This function sets default parameters for the A-KAZE detector
  void setDefaultAKAZEOptions(Options& options);




  /// This function checks descriptor limits for a given keypoint
  inline void check_descriptor_limits(int& x, int& y, int width, int height);

  /// This function computes the value of a 2D Gaussian function
  inline float gaussian(float x, float y, float sigma) {
    return expf(-(x*x+y*y)/(2.0f*sigma*sigma));
  }

  /// This funtion rounds float to nearest integer
  inline int fRound(float flt) {
    return (int)(flt+0.5f);
  }

  struct AKAZE_EXPORT CudaDevice
  {
      int id;
      std::string name;
  };

  AKAZE_EXPORT std::vector<CudaDevice> getDevices();

  //AKAZE_EXPORT void openDevice();
  //AKAZE_EXPORT void openDevice(std::string deviceName);
  //AKAZE_EXPORT void openDevice(std::string platform, std::string deviceName);

}}//namspace libAKAZE::cuda

#endif //AKAZE_USE_CUDA
