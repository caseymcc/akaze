//=============================================================================
//
// akaze_features.cpp
// Authors: Pablo F. Alcantarilla (1), Jesus Nuevo (2)
// Institutions: Toshiba Research Europe Ltd (1)
//               TrueVision Solutions (2)
// Date: 07/10/2014
// Email: pablofdezalc@gmail.com
//
// AKAZE Features Copyright 2014, Pablo F. Alcantarilla, Jesus Nuevo
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file akaze_features.cpp
 * @brief Main program for detecting and computing binary descriptors in an
 * accelerated nonlinear scale space
 * @date Oct 07, 2014
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#include "commandLineHelper.h"

#include <ctime>

#include "lib/AKAZE.h"
#include "lib/AKAZEConfig.h"
#include "lib/utils.h"
#include "timer/timer.hpp"

#include "cimg/CImg.h"

#ifdef AKAZE_USE_CUDA
#include "./lib/AKAZE_cuda.h"
#include <cuda_profiler_api.h>
#endif //AKAZE_USE_CUDA

#ifdef AKAZE_USE_OPENCL
#include "./lib/OpenCLContext.h"
#include "./lib/AKAZE_cl.h"
#endif

using namespace std;

int featuresStandard(ProgramOptions &options, libAKAZE::Options &akaze_options);
int featuresCuda(ProgramOptions &options, libAKAZE::Options &akaze_options);
int featuresOpenCL(ProgramOptions &options, libAKAZE::Options &akaze_options);

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
    // Variables
    ProgramOptions options;
    libAKAZE::Options akaze_options;
    std::string image_path, keypoints_path;
    
    int error;

    // Parse the input command line options
    if((error=parse_input_options(options, akaze_options, argc, argv)))
    {
        if(error < 0)
            show_input_options_help(0);

#if(defined(_WINDOWS) && defined(_DEBUG))
        system("pause");
#endif
        return -1;
    }

    if(akaze_options.verbosity)
    {
        cout<<"Check AKAZE options:"<<endl;
        cout<<akaze_options<<endl;
    }

    if(options.requestDevices)
    {
        if(akaze_options.type==libAKAZE::Options::Cuda)
        {
            int retVal=0;

#ifdef AKAZE_USE_CUDA
            std::vector<libAKAZE::cuda::CudaDevice> devices=libAKAZE::cuda::getDevices();

            for(libAKAZE::cuda::CudaDevice &device:devices)
            {
                cout<<"Device: "<<device.name<<endl;
            }
#else
            cerr<<"Error: Cuda not supported in build"<<endl;
            retVal=-1;
#endif
#if(defined(_WINDOWS) && defined(_DEBUG))
            system("pause");
#endif
            return retVal;
        }
        else if(akaze_options.type==libAKAZE::Options::OpenCL)
        {
            int retVal=0;
#ifdef AKAZE_USE_OPENCL
            std::vector<libAKAZE::cl::OpenClDevice> devices=libAKAZE::cl::getDevices();

            for(libAKAZE::cl::OpenClDevice &device:devices)
            {
                cout<<"Device: "<<device.name<<" ("<<device.platform<<", "<<device.vendor<<")"<<endl;
                cout<<"  Type: "<<((device.type==libAKAZE::cl::OpenClDevice::GPU)?"GPU":"CPU")<<endl;
                cout<<"  Version: "<<device.version<<endl;
            }
#else
            cerr<<"Error: OpenCL not supported in build"<<endl;
            retVal=-1;
#endif // AKAZE_USE_OPENCL

#if(defined(_WINDOWS) && defined(_DEBUG))
            system("pause");
#endif
            return retVal;
        }
        else
        {
            cerr<<"Error: --devices not supported for Standard processing"<<endl;

#if(defined(_WINDOWS) && defined(_DEBUG))
            system("pause");
#endif
            return -1;
        }
    }

    // Try to read the image and if necessary convert to grayscale.
    int value;

    switch(akaze_options.type)
    {
    case libAKAZE::Options::Standard:
        value=featuresStandard(options, akaze_options);
        break;
    case libAKAZE::Options::Cuda:
        value=featuresCuda(options, akaze_options);
        break;
    case libAKAZE::Options::OpenCL:
        value=featuresOpenCL(options, akaze_options);
        break;
    }

#if(defined(_WINDOWS) && defined(_DEBUG))
    system("pause");
#endif
    return value;
}

int featuresStandard(ProgramOptions &options, libAKAZE::Options &akaze_options)
{
    // Try to read the image and if necessary convert to grayscale. CImg will
    // throw an error and crash if the image could not be read.
    cimg_library::CImg<float> img(options.image_path.c_str());
    RowMatrixXf img_32;
    timer::Timer timer;
    double totalTime;

    ConvertCImgToEigen(img, img_32);
    img_32/=255.0;

    // Don't forget to specify image dimensions in AKAZE's options.
    akaze_options.img_width=img_32.cols();
    akaze_options.img_height=img_32.rows();

    // Extract features.
    std::vector<libAKAZE::Keypoint> kpts;
    libAKAZE::AKAZE evolution(akaze_options);
    libAKAZE::Descriptors desc;

    timer.reset();
    evolution.Create_Nonlinear_Scale_Space(img_32);
    evolution.Feature_Detection(kpts);

    // Compute descriptors.
    evolution.Compute_Descriptors(kpts, desc);
    totalTime=timer.elapsedMs();

    // Summarize the computation times.
    evolution.Save_Scale_Space();
    std::cout<<"Number of points: "<<kpts.size()<<std::endl;
    evolution.Show_Computation_Times();
    std::cout<<"Total Time: "<<totalTime<<std::endl;

    // Save keypoints in ASCII format.
    if(akaze_options.descriptor < libAKAZE::MLDB_UPRIGHT)
    {
        save_keypoints(options.keypoints_path, kpts, desc, true);
    }
    else
    {
//            save_keypoints(options.keypoints_path, kpts, desc.binary_descriptor, true);
        save_keypoints_json(options.keypoints_path, kpts, desc, true);
    }

    // Convert the input image to RGB.
    cimg_library::CImg<float> rgb_image=
        img.get_resize(img.width(), img.height(), img.depth(), 3);
//    draw_keypoints(rgb_image, kpts);
    draw_keypoints_vector(rgb_image, kpts);
    rgb_image.save("../output/detected_features.jpg");

    return 0;
}

int featuresCuda(ProgramOptions &options, libAKAZE::Options &akaze_options)
{
#ifdef AKAZE_USE_CUDA
    cimg_library::CImg<float> img(options.image_path.c_str());
    RowMatrixXf img_32;
    timer::Timer timer;
    double totalTime;

    ConvertCImgToEigen(img, img_32);
    img_32/=255.0;

    akaze_options.img_width=img_32.cols();
    akaze_options.img_height=img_32.rows();

    if(!options.device.empty())
    {
    }

    // Extract features
    libAKAZE::cuda::AKAZE evolution(akaze_options);
    vector<libAKAZE::Keypoint> kpts;
    libAKAZE::Descriptors desc;

//    cudaProfilerStart();
    timer.reset();
    evolution.Create_Nonlinear_Scale_Space(img_32);
    evolution.Feature_Detection(kpts);
    
    // Compute descriptors.
    evolution.Compute_Descriptors(kpts, desc);
    totalTime=timer.elapsedMs();

//    cudaProfilerStop();

    std::cout<<"Number of points: "<<kpts.size()<<std::endl;
    evolution.Show_Computation_Times();
    std::cout<<"Total Time: "<<totalTime<<std::endl;

    std::string fileName="../output/keypoints_cuda.txt";
    save_keypoints_json(fileName, kpts, desc, true);

    cimg_library::CImg<float> rgb_image=
        img.get_resize(img.width(), img.height(), img.depth(), 3);
    draw_keypoints_vector(rgb_image, kpts);
    rgb_image.save("../output/detected_features_cuda.jpg");

    return 0;
#else
    return -1;
#endif
}

int featuresOpenCL(ProgramOptions &options, libAKAZE::Options &akaze_options)
{
#ifdef AKAZE_USE_OPENCL
    cimg_library::CImg<float> img(options.image_path.c_str());
    RowMatrixXf img_32;
    timer::Timer timer;
    double totalTime;

    ConvertCImgToEigen(img, img_32);
    img_32/=255.0;

    akaze_options.img_width=img_32.cols();
    akaze_options.img_height=img_32.rows();

    ::cl::Context openClContext;
    libAKAZE::cl::OpenClDevice deviceInfo;

    if(!options.device.empty())
    {
        if(!options.platform.empty())
            openClContext=libAKAZE::cl::openDevice(options.platform, options.device, deviceInfo);
        else
            openClContext=libAKAZE::cl::openDevice(options.device, deviceInfo);
    }
    else
    {
        openClContext=libAKAZE::cl::openDevice(deviceInfo);
    }

    if(openClContext()==nullptr)
        return 1;

    cout<<"Device: "<<deviceInfo.name<<" ("<<deviceInfo.platform<<", "<<deviceInfo.vendor<<")"<<endl;
    cout<<"  Type: "<<((deviceInfo.type==libAKAZE::cl::OpenClDevice::GPU)?"GPU":"CPU")<<endl;
    cout<<"  Version: "<<deviceInfo.version<<endl;

    ::cl::CommandQueue commandQueue(openClContext);

    vector<libAKAZE::Keypoint> kpts;
    libAKAZE::cl::AKAZE evolution(openClContext, commandQueue, akaze_options);
    libAKAZE::Descriptors desc;
        
    evolution.initOpenCL(); //gets some items loaded early
        
    timer.reset();
    if(!options.scale_space_directory.empty()&&!options.save_scale_space)
        evolution.Load_Nonlinear_Scale_Space(options.scale_space_directory);
    else
    {
        evolution.Create_Nonlinear_Scale_Space(img_32);
        
        if(options.save_scale_space)
            evolution.Save_Nonlinear_Scale_Space(options.scale_space_directory);
    }
        
    evolution.Feature_Detection(kpts);
    //    if(!options.det_hessian_response_directory.empty()&&!options.save_det_hessian_response)
    //    {
    //        evolution.Load_Derivatives(options.det_hessian_response_directory);
    //        evolution.Load_Determinant_Hessian_Response(options.det_hessian_response_directory);
    //    }
    //    else
    //    {
    //        evolution.Compute_Determinant_Hessian_Response();
    //
    //        if(options.save_det_hessian_response)
    //        {
    //            evolution.Save_Derivatives(options.det_hessian_response_directory);
    //            evolution.Save_Determinant_Hessian_Response(options.det_hessian_response_directory);
    //        }
    //    }
    //
    //    if(!options.keypoints_data_path.empty() && !options.save_keypoints)
    //    {
    //        evolution.Load_Keypoints(options.keypoints_data_path);
    //    }
    //    else
    //    {
    ////        evolution.Find_Scale_Space_Extrema(kpts);
    //        evolution.Find_Scale_Space_Extrema();
    //
    //        if(options.save_keypoints)
    //            evolution.Save_Keypoints(options.keypoints_data_path);
    //    }
        
    evolution.Compute_Descriptors(desc);
        
    totalTime=timer.elapsedMs();
        
    evolution.getKeypoints(kpts);
        
    std::cout<<"Number of points: "<<kpts.size()<<std::endl;
    evolution.Show_Computation_Times();
    std::cout<<"Total Time: "<<totalTime<<std::endl;
        
    std::string fileName="../output/keypoints_cl.txt";
    save_keypoints_json(fileName, kpts, desc, true);
        
    cimg_library::CImg<float> rgb_image=
        img.get_resize(img.width(), img.height(), img.depth(), 3);
    draw_keypoints_vector(rgb_image, kpts);
    rgb_image.save("../output/detected_features_cl.jpg");

    return 0;
#else //AKAZE_USE_OPENCL
    return -1;
#endif //AKAZE_USE_OPENCL
}