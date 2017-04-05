//=============================================================================
//
// bechmark.cpp
//=============================================================================

/**
 * @file benchmark.cpp
 * @brief Used for benchmarking AKAZE library functions
 * @date 04/05/2017
 * @author Casey McCandless
 */

#include <gflags/gflags.h>
#include <ctime>

#include "lib/AKAZE.h"
#include "lib/AKAZEConfig.h"
#include "lib/utils.h"
#include "timer/timer.hpp"
#include "cimg/CImg.h"

#include "benchmark_options.h"

#ifdef AKAZE_USE_CUDA
#include "./lib/AKAZE_cuda.h"
#include <cuda_profiler_api.h>
#endif //AKAZE_USE_CUDA

#ifdef AKAZE_USE_OPENCL
//#include "./lib/OpenCLContext.h"
//#include "./lib/AKAZE_cl.h"
#include "benchmark_opencl.h"
#endif

void listDevices();
void benchmarkCPU(RowMatrixXf image, Options &options);
void benchmarkOpenCL(RowMatrixXf image, Options &options);
void benchmarkCUDA(RowMatrixXf image, Options &options);

DEFINE_bool(devices, false, "List all available processing devices");
DEFINE_bool(disable_cpu, false, "Benchmarks cpu based functions");
DEFINE_bool(opencl, false, "Benchmarks opencl based functions");
DEFINE_string(opencl_platform, "any", "OpenCL platform to use for benchmarking");
DEFINE_string(opencl_device, "any", "OpenCL device to use for benchmarking");
DEFINE_bool(cuda, false, "Benchmarks cuda based functions");

DEFINE_string(input, "benchmark.jpg", "Input image used for benchmarks");
DEFINE_int32(iter, 100, "Number of runs to average results over");
DEFINE_bool(save_images, false, "Tells benchmark to save test images");


Options getOptions()
{
    Options options;

    options.iterations=FLAGS_iter;
    options.saveImages=FLAGS_save_images;
    return options;
}

/* ************************************************************************* */
/**
 * @brief This function parses the command line arguments for setting A-KAZE
 * parameters
 * @param options Structure that contains A-KAZE settings
 * @param img_path Path for the input image
 * @param kpts_path Path for the file where the keypoints where be stored
 */
//int parse_input_options(libAKAZE::Options& options, std::string& img_path,
//                        std::string& kpts_path, int argc, char* argv[]);

/* ************************************************************************* */
int main(int argc, char* argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if(FLAGS_devices)
    {
        listDevices();
    }
    else
    {
        cimg_library::CImg<float> img(FLAGS_input.c_str());
        RowMatrixXf img_32;

        ConvertCImgToEigen(img, img_32);
        img_32/=255.0;

        Options options=getOptions();

        if(!FLAGS_disable_cpu)
            benchmarkCPU(img_32, options);
        if(FLAGS_opencl)
            benchmarkOpenCL(img_32, options);
        if(FLAGS_cuda)
            benchmarkCUDA(img_32, options);
    }

#if(defined(_WINDOWS))// && defined(_DEBUG))
    system("pause");
#endif
    return 0;
}

void listDevices()
{
#ifdef AKAZE_USE_OPENCL
    std::vector<libAKAZE::cl::OpenClDevice> openclDevices=libAKAZE::cl::getDevices();

    if(!openclDevices.empty())
    {
        std::cout<<"OpenCL devices:"<<std::endl;

        for(libAKAZE::cl::OpenClDevice &openclDevice:openclDevices)
        {
            std::cout<<"  Device: "<<openclDevice.name<<" ("<<openclDevice.platform<<", "<<openclDevice.vendor<<")"<<std::endl;
            std::cout<<"    Type: "<<((openclDevice.type==libAKAZE::cl::OpenClDevice::GPU)?"GPU":"CPU")<<std::endl;
            std::cout<<"    Version: "<<openclDevice.version<<std::endl;
        }
        std::cout<<std::endl;
    }
#endif //AKAZE_USE_OPENCL
#ifdef AKAZE_USE_CUDA
    std::vector<libAKAZE::cuda::CudaDevice> cudaDevices=libAKAZE::cuda::getDevices();

    if(!cudaDevices.empty())
    {
        std::cout<<"CUDA devices:"<<std::endl;

        for(libAKAZE::cuda::CudaDevice &cudaDevice:cudaDevices)
        {
            std::cout<<"  Device: "<<cudaDevice.name<<std::endl;
        }
    }
#endif //AKAZE_USE_CUDA
}

void benchmarkCPU(RowMatrixXf image, Options &options)
{

}

void benchmarkOpenCL(RowMatrixXf image, Options &options)
{
#ifdef AKAZE_USE_OPENCL
    ::cl::Context openClContext;
    libAKAZE::cl::OpenClDevice deviceInfo;

    if(FLAGS_opencl_device=="any")
    {
        openClContext=libAKAZE::cl::openDevice(deviceInfo);
    }
    else
    {
        if(FLAGS_opencl_platform == "any")
            openClContext=libAKAZE::cl::openDevice(FLAGS_opencl_device, deviceInfo);
        else
            openClContext=libAKAZE::cl::openDevice(FLAGS_opencl_platform, FLAGS_opencl_device, deviceInfo);
    }

    if(openClContext() == nullptr)
        return;

    std::cout<<"Device: "<<deviceInfo.name<<" ("<<deviceInfo.platform<<", "<<deviceInfo.vendor<<")"<<std::endl;
    std::cout<<"  Type: "<<((deviceInfo.type==libAKAZE::cl::OpenClDevice::GPU)?"GPU":"CPU")<<std::endl;
    std::cout<<"  Version: "<<deviceInfo.version<<std::endl;
#endif //AKAZE_USE_OPENCL

    benchmarkOpenCLDevice(openClContext, image, options);
}

void benchmarkCUDA(RowMatrixXf image, Options &options)
{

}