#ifndef _convolve_cl_h_
#define _convolve_cl_h_

#include "CL/cl.hpp"
#include "AKAZEConfig.h"

namespace libAKAZE{namespace cl
{

///
/// Performs convolution on serparable kernels
///
void separableConvolve(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, ::cl::Buffer kernelX, int kernelXSize, ::cl::Buffer kernelY, int kernelYSize, ::cl::Image2D &dst);
//void separableConvolve(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, ::cl::Image2D &scratch, ::cl::Image2D &dst, size_t width, size_t height, 
//    ::cl::Buffer kernelX, int kernelXSize, ::cl::Buffer kernelY, int kernelYSize, float scale, std::vector<::cl::Event> *events, ::cl::Event &event);
void separableConvolve(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, size_t width, size_t height, ::cl::Buffer kernelXBuffer, int kernelXSize,
    ::cl::Buffer kernelYBuffer, int kernelYSize, float scale, ::cl::Image2D &dst, ::cl::Image2D &scratch, std::vector<::cl::Event> *events, ::cl::Event &event);
void separableConvolve(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, size_t width, size_t height, ::cl::Buffer kernelXBuffer, int kernelXSize,
    ::cl::Buffer kernelYBuffer, int kernelYSize, float scale, ::cl::Buffer &dst, size_t offset, ::cl::Image2D &scratch, std::vector<::cl::Event> *events, ::cl::Event &event);
void separableConvolve(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Buffer &src, size_t srcOffset, size_t width, size_t height, ::cl::Buffer kernelXBuffer, int kernelXSize,
    ::cl::Buffer kernelYBuffer, int kernelYSize, float scale, ::cl::Buffer &dst, size_t dstOffset, ::cl::Buffer &scratch, std::vector<::cl::Event> *events, ::cl::Event &event);

///
/// Constructs guassian kernel and copies it to a cl::Buffer object that is retuned.
///
::cl::Buffer buildGaussianKernel(::cl::Context context, ::cl::CommandQueue commandQueue, float sigma, int &filterSize);

///
/// Performs guassian convolution, will be cl::Buffer internally
///
void gaussianConvolution(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, ::cl::Image2D &dst, size_t width, size_t height, float sigma, std::vector<::cl::Event> *events, ::cl::Event &event);

///
/// Construct guassian kernel as 2 kernels that work on the x/y direction.
///
::cl::Buffer buildGaussianSeparableKernel(::cl::Context context, ::cl::CommandQueue commandQueue, float sigma, int &filterSize);

///
///
///
void gaussianSeparable(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, ::cl::Image2D &dst, size_t width, size_t height, float sigma, std::vector<::cl::Event> *events, ::cl::Event &event);
void gaussianSeparable(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, ::cl::Image2D &dst, size_t width, size_t height, ::cl::Buffer kernelBuffer, int kernelSize, ::cl::Image2D scratch,
    std::vector<::cl::Event> *events, ::cl::Event &event);
//double lowPass(CudaImage &inimg, CudaImage &outimg, CudaImage &temp, double var, int kernsize);

struct ScharrSeparableKernel
{
    ::cl::Buffer smooth;
    ::cl::Buffer edge;
};

///
///
///
ScharrSeparableKernel buildScharrSeparableKernel(::cl::Context context, int scale, int &kernelSize, bool normalize);

///
///
///
void scharrSeparable(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, size_t width, size_t height, int size, float scale, bool yKernel, bool normalize, ::cl::Image2D &dst, std::vector<::cl::Event> *events, ::cl::Event &event);
void scharrSeparable(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, size_t width, size_t height, int size, float scale, bool yKernel, bool normalize, ::cl::Image2D &dst,
    ScharrSeparableKernel &kernelBuffer, int kernelSize, ::cl::Image2D scratch, std::vector<::cl::Event> *events, ::cl::Event &event);

void scharrSeparable(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, size_t width, size_t height, int size, float scale, bool yKernel, bool normalize, ::cl::Buffer &dst, size_t offset, 
    std::vector<::cl::Event> *events, ::cl::Event &event);
void scharrSeparable(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Buffer &src, size_t srcOffset, size_t width, size_t height, int size, float scale, bool yKernel, bool normalize,
    ::cl::Buffer &dst, size_t dstOffset, std::vector<::cl::Event> *events, ::cl::Event &event);

}}//namespace libAKAZE::cl

#endif //_convolve_cl_h_