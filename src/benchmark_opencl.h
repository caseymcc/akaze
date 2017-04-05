#ifndef _benchmark_opencl_h_
#define _benchmark_opencl_h_

#include "lib/AKAZEConfig.h"

#ifdef AKAZE_USE_OPENCL

#include "./lib/OpenCLContext.h"
#include "./lib/AKAZE_cl.h"
#include "cimg/CImg.h"
#include "benchmark_options.h"

void benchmarkOpenCLDevice(::cl::Context &openClContext, RowMatrixXf &image, Options &options);

#endif //AKAZE_USE_OPENCL

#endif //_benchmark_opencl_h_