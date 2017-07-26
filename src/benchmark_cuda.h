#ifndef _benchmark_cuda_h_
#define _benchmark_cuda_h_

#include "lib/AKAZEConfig.h"

#ifdef AKAZE_USE_CUDA

#include "./lib/AKAZE_cuda.h"
#include "cimg/CImg.h"
#include "benchmark_options.h"

void benchmarkCudaDevice(openClContext, RowMatrixXf &image, Options &options);

#endif //AKAZE_USE_OPENCL

#endif //_benchmark_opencl_h_