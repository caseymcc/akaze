/**
 * @file AKAZE_cl.h
 */

#pragma once

#include "AKAZE.h"

#ifdef AKAZE_USE_OPENCL

#include <CL/cl.hpp>
#include <memory>
#include "filters_cl.h"

/* ************************************************************************* */
namespace libAKAZE
{
namespace cl
{

#ifdef _MSC_VER
struct alignas(16) Keypoint
#else
struct __attribute__((aligned(16))) Keypoint
#endif
{
    int class_id;
    int octave;
#ifdef TRACK_REMOVED
    int removed;
#endif //TRACK_REMOVED
    float size;
    float ptX;
    float ptY;
    float response;
    float angle;    
};

#ifdef _MSC_VER
struct alignas(16) EvolutionInfo
#else
struct __attribute__((aligned(16))) EvolutionInfo
#endif
{
    int width;
    int height;
    int offset;
    float sigma;
    float pointSize;
    int octave;
};

#ifdef _MSC_VER
struct alignas(8) ExtremaMap
#else
struct __attribute__((aligned(8))) ExtremaMap
#endif
{
    int class_id;
    float response;
};

struct EvolutionCL:Evolution
{
    size_t offset;

    ::cl::Image2D image;              ///< Evolution image
    ::cl::Image2D smooth;             ///< Smoothed image
    ::cl::Image2D lx, ly;             ///< First order spatial derivatives
    ::cl::Image2D flow;               ///< Diffusivity image
    ::cl::Image2D step;               ///< Evolution step update
    ::cl::Image2D scratch;

    ::cl::Buffer dx; //First order spatial derivatives
    ::cl::Buffer dy; //First order spatial derivatives

    ::cl::Buffer dxx; //Second order spatial derivatives
    ::cl::Buffer dxy; //Second order spatial derivatives
    ::cl::Buffer dyy; //Second order spatial derivatives

    ::cl::Buffer det; //Detector response

    ::cl::Image2D scratchBuffer;
};

struct KernelInfo;
typedef std::shared_ptr<KernelInfo> SharedKernelInfo;

class AKAZE_EXPORT AKAZE//:public libAKAZE::AKAZE
{

public:
    /// AKAZE constructor with input options
    /// @param options AKAZE configuration options
    /// @note This constructor allocates memory for the nonlinear scale space
//    AKAZE(cl_context openclContext, cl_command_queue commandQueue, const Options& options);
    AKAZE(::cl::Context openclContext, ::cl::CommandQueue commandQueue, const Options& options);

    /// Destructor
    ~AKAZE();

    void initOpenCL();

    int Create_Nonlinear_Scale_Space(const RowMatrixXf &img);
    void Feature_Detection(std::vector<libAKAZE::Keypoint> &kpts);
    void Feature_Detection();

    void Load_Nonlinear_Scale_Space(std::string &directory);
    void Save_Nonlinear_Scale_Space(std::string &directory);

    void Load_Derivatives(std::string &directory);
    void Save_Derivatives(std::string &directory);

    void Load_Determinant_Hessian_Response(std::string &directory);
    void Save_Determinant_Hessian_Response(std::string &directory);

    void Compute_Determinant_Hessian_Response();
    void Find_Scale_Space_Extrema(std::vector<libAKAZE::Keypoint> &kpts);
    void Find_Scale_Space_Extrema();

    void Compute_Descriptors(std::vector<libAKAZE::Keypoint> &kpts, Descriptors &desc);
    void Compute_Descriptors(Descriptors &desc);
    void Compute_Descriptors();

    void Load_Keypoints(std::string fileName);
    void Save_Keypoints(std::string fileName);

    void getKeypoints(std::vector<libAKAZE::Keypoint> &kpts);
    void putKeypoints(std::vector<libAKAZE::Keypoint> &kpts);

    void getDescriptors(Descriptors &desc);

	void saveDebug();

    void Show_Computation_Times() const;
private:
    void Allocate_Memory_Evolution();
    void Compute_Multiscale_Derivatives(std::vector<std::vector<::cl::Event>> &evolutionEvents);

    void Compute_Main_Orientation();
//    void Do_Subpixel_Refinement(std::vector<Keypoint> &kpts);
    void Get_MLDB_Full_Descriptor();

    Options options_;
    Timing timing_;

    std::vector<EvolutionCL> evolution_;         ///< Vector of nonlinear diffusion evolution
    ::cl::Buffer evolutionInfo_;

    int ncycles_;                               ///< Number of cycles
    bool reordering_;                           ///< Flag for reordering time steps
    std::vector<std::vector<float > > tsteps_;  ///< Vector of FED dynamic time steps
    std::vector<int> nsteps_;                   ///< Vector of number of steps per cycle

//    cl_context openclContext_;
    ::cl::Context openclContext_;
    ::cl::CommandQueue commandQueue_;

    ::cl::Image2D inputGuassian_;
    size_t width_;
    size_t height_;

    bool saveImages_;
    bool saveCsv_;

    ::cl::Buffer evolutionImage_;
    ::cl::Buffer evolutionDx_;
    ::cl::Buffer evolutionDy_;
    ::cl::Buffer evolutionDxx_;
    ::cl::Buffer evolutionDxy_;
    ::cl::Buffer evolutionDyy_;
    ::cl::Buffer evolutionDet_;

    ::cl::Buffer offsetGuassian_;
    int offsetGuassianSize_;

    ::cl::Image2D contrastGuassianScratch_;
    ::cl::Image2D contrastMagnitudeScratch_;
    ::cl::Buffer guassian_1_0_;
    int guassianSize_1_0_;
    ScharrSeparableKernel scharr_1_0_;
    int scharrSize_1_0_;
    ::cl::Buffer histogramBuffer_;
    std::vector<int> histogram_;
    ::cl::Buffer histogramScratchBuffer_;
    ::cl::Buffer maxBuffer_;
    ::cl::Buffer contrastBuffer_;

    ::cl::Buffer keypointsBuffer_;
    int keypointsCount_;

    ::cl::Buffer descriptorsBuffer_;
    size_t descriptorBufferSize_;
    size_t descriptorSize_;

    ::cl::Buffer extremaMap_;
    
    SharedKernelInfo contrastKernel_;
};

}}//namespace libAkaze::cl

#endif //AKAZE_USE_OPENCL
