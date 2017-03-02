#include "filters_cl.h"
#include "nldiffusion_functions.h"

#include "openClContext.h"

#include "AKAZE_cl.h"
#include <fstream>

namespace libAKAZE
{
namespace cl
{

void separableConvolve(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, ::cl::Buffer kernelXBuffer, int kernelXSize, ::cl::Buffer kernelYBuffer, int kernelYSize, ::cl::Image2D &dst)
{
    ::cl::Event event;

    size_t width, height;
    cl_image_format format;
        
    src.getImageInfo(CL_IMAGE_WIDTH, &width);
    src.getImageInfo(CL_IMAGE_HEIGHT, &height);
    src.getImageInfo(CL_IMAGE_FORMAT, &format);

    ::cl::Image2D scratch(context, CL_MEM_READ_WRITE, ::cl::ImageFormat(format.image_channel_order, format.image_channel_data_type), width, height);

    separableConvolve(context, commandQueue, src, width, height, kernelXBuffer, kernelXSize, kernelYBuffer, kernelYSize, 1.0, dst, scratch, nullptr, event);
    event.wait();
}

void separableConvolve(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, size_t width, size_t height, ::cl::Buffer kernelXBuffer, int kernelXSize, 
    ::cl::Buffer kernelYBuffer, int kernelYSize, float scale, ::cl::Image2D &dst, ::cl::Image2D &scratch, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    size_t filterSize;

    ::cl::Kernel kernelX=getKernel(context, "separableConvolveXImage2D", "lib/kernels/convolve.cl");
    ::cl::Kernel kernelY=getKernel(context, "separableConvolveYImage2D", "lib/kernels/convolve.cl");
    cl_int status;
    int index=0;
    ::cl::Event kernelYEvent;

    status=kernelY.setArg(index++, src);
    status=kernelY.setArg(index++, kernelYBuffer);
    status=kernelY.setArg(index++, kernelYSize);
    status=kernelY.setArg(index++, (float)1.0); //only scale once, so no scale here
    status=kernelY.setArg(index++, scratch);

    ::cl::NDRange globalThreads(width, height);

    status=commandQueue.enqueueNDRangeKernel(kernelY, ::cl::NullRange, globalThreads, ::cl::NullRange, events, &kernelYEvent);

    std::vector<::cl::Event> kernelYEvents={kernelYEvent};
    index=0;
    
    status=kernelX.setArg(index++, scratch);
    status=kernelX.setArg(index++, kernelXBuffer);
    status=kernelX.setArg(index++, kernelXSize);
    status=kernelX.setArg(index++, scale);
    status=kernelX.setArg(index++, dst);

    status=commandQueue.enqueueNDRangeKernel(kernelX, ::cl::NullRange, globalThreads, ::cl::NullRange, &kernelYEvents, &event);

    commandQueue.flush();
}

void separableConvolve(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, size_t width, size_t height, ::cl::Buffer kernelXBuffer, int kernelXSize, 
    ::cl::Buffer kernelYBuffer, int kernelYSize, float scale, ::cl::Buffer &dst, size_t offset, ::cl::Image2D &scratch, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    size_t filterSize;

    ::cl::Kernel kernelX=getKernel(context, "separableConvolveXImage2DBuffer", "lib/kernels/convolve.cl");
    ::cl::Kernel kernelY=getKernel(context, "separableConvolveYImage2D", "lib/kernels/convolve.cl");
    cl_int status;
    int index=0;
    ::cl::Event kernelYEvent;

    status=kernelY.setArg(index++, src);
    status=kernelY.setArg(index++, kernelYBuffer);
    status=kernelY.setArg(index++, kernelYSize);
    status=kernelY.setArg(index++, (float)1.0); //only scale once, so no scale here
    status=kernelY.setArg(index++, scratch);

    ::cl::NDRange globalThreads(width, height);

    status=commandQueue.enqueueNDRangeKernel(kernelY, ::cl::NullRange, globalThreads, ::cl::NullRange, events, &kernelYEvent);

    std::vector<::cl::Event> kernelYEvents={kernelYEvent};
    index=0;

    //(read_only image2d_t input, __constant float *kernelX, const int kernelSize, float scale, __global float *output, int offset, int width, int height)

    status=kernelX.setArg(index++, scratch);
    status=kernelX.setArg(index++, kernelXBuffer);
    status=kernelX.setArg(index++, kernelXSize);
    status=kernelX.setArg(index++, scale);
    status=kernelX.setArg(index++, dst);
    status=kernelX.setArg(index++, (int)offset);
    status=kernelX.setArg(index++, (int)width);
    status=kernelX.setArg(index++, (int)height);

    status=commandQueue.enqueueNDRangeKernel(kernelX, ::cl::NullRange, globalThreads, ::cl::NullRange, &kernelYEvents, &event);

    commandQueue.flush();
}

void separableConvolve(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Buffer &src, size_t srcOffset, size_t width, size_t height, ::cl::Buffer kernelXBuffer, int kernelXSize,
    ::cl::Buffer kernelYBuffer, int kernelYSize, float scale, ::cl::Buffer &dst, size_t dstOffset, ::cl::Buffer &scratch, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    size_t filterSize;

    ::cl::Kernel kernelX=getKernel(context, "separableConvolveXBuffer", "lib/kernels/convolve.cl");
    ::cl::Kernel kernelY=getKernel(context, "separableConvolveYBuffer", "lib/kernels/convolve.cl");
    cl_int status;
    int index=0;
    ::cl::Event kernelYEvent;

    status=kernelY.setArg(index++, src);
    status=kernelY.setArg(index++, (int)srcOffset);
    status=kernelY.setArg(index++, (int)width);
    status=kernelY.setArg(index++, (int)height);
    status=kernelY.setArg(index++, kernelYBuffer);
    status=kernelY.setArg(index++, kernelYSize);
    status=kernelY.setArg(index++, (float)1.0); //only scale once, so no scale here
    status=kernelY.setArg(index++, scratch);
    status=kernelY.setArg(index++, (int)0);

    ::cl::NDRange globalThreads(width, height);

    status=commandQueue.enqueueNDRangeKernel(kernelY, ::cl::NullRange, globalThreads, ::cl::NullRange, events, &kernelYEvent);

    std::vector<::cl::Event> kernelYEvents={kernelYEvent};
    
    index=0;
    status=kernelX.setArg(index++, scratch);
    status=kernelX.setArg(index++, (int)0);
    status=kernelX.setArg(index++, (int)width);
    status=kernelX.setArg(index++, (int)height);
    status=kernelX.setArg(index++, kernelXBuffer);
    status=kernelX.setArg(index++, kernelXSize);
    status=kernelX.setArg(index++, scale);
    status=kernelX.setArg(index++, dst);
    status=kernelX.setArg(index++, (int)dstOffset);

    status=commandQueue.enqueueNDRangeKernel(kernelX, ::cl::NullRange, globalThreads, ::cl::NullRange, &kernelYEvents, &event);

    commandQueue.flush();
}

::cl::Buffer buildGaussianKernel(::cl::Context context, ::cl::CommandQueue commandQueue, float sigma, int &filterSize)
{
    int size=(int)ceil((1.0+(sigma-0.8)/(0.3)));
    float sum=0.0f;
    float twoSigmaSquared=2*sigma*sigma;
    
    filterSize=2*size+1;
    float *filter=(float *)malloc(filterSize*filterSize*sizeof(float));

    for(int x=-size; x < size+1; x++)
    {
        for(int y=-size; y < size+1; y++)
        {
            float temp=exp(-((float)(x*x+y*y)/twoSigmaSquared));

            sum+=temp;
            filter[x+size+(y+size)*(filterSize)]=temp;
        }
    }
    
    //normalize filter
    for(int i=0; i < filterSize*filterSize; i++)
        filter[i]=filter[i]/sum;

//    ::cl::Buffer filterBuffer(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, sizeof(cl_float)*filterSize*filterSize, filter);
    ::cl::Buffer filterBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*filterSize*filterSize, filter);
    
//    commandQueue.enqueueWriteBuffer(filterBuffer, true, 0, )
    
    free(filter);

    return filterBuffer;
}

::cl::Buffer buildGaussianSeparableKernel(::cl::Context context, ::cl::CommandQueue commandQueue, float sigma, int &kernelSize)
{
    int size=(int)ceil((1.0+(sigma-0.8)/(0.3)));
    float sum=0.0f;
    float twoSigmaSquared=2*sigma*sigma;

    kernelSize=2*size+1;
    float *kernel=(float *)malloc(kernelSize*sizeof(float));
    float temp;

    int x=-size;
    for(int i=0; i<kernelSize; ++i, ++x)
    {
        temp=std::exp(-(float)(x*x)/twoSigmaSquared);
        
        kernel[i]=temp;
        sum+=temp;
    }

    //normalize filter
    for(int i=0; i < kernelSize; i++)
        kernel[i]=kernel[i]/sum;

    ::cl::Buffer kernelBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*kernelSize, kernel);
    free(kernel);

    return kernelBuffer;
}

void gaussianConvolution(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, ::cl::Image2D &dst, size_t width, size_t height, float sigma, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    int kernelSize;// =computeKernelSize(sigma);

    ::cl::Buffer kernelBuffer=buildGaussianKernel(context, commandQueue, sigma, kernelSize);

    ::cl::Kernel kernel=getKernel(context, "convolve", "lib/kernels/convolve.cl");
    cl_int status;
    int index=0;

    status=kernel.setArg(index++, src);
    status=kernel.setArg(index++, kernelBuffer);
    status=kernel.setArg(index++, kernelSize);
    status=kernel.setArg(index++, dst);

    ::cl::NDRange globalThreads(width, height);

    status=commandQueue.enqueueNDRangeKernel(kernel, ::cl::NullRange, globalThreads, ::cl::NullRange, events, &event);

    commandQueue.flush();
//    event.wait();
}

void gaussianSeparable(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, ::cl::Image2D &dst, size_t width, size_t height, float sigma, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    int kernelSize;

    ::cl::Buffer kernelBuffer=buildGaussianSeparableKernel(context, commandQueue, sigma, kernelSize);

//    size_t width, height;
    cl_image_format format;

//    src.getImageInfo(CL_IMAGE_WIDTH, &width);
//    src.getImageInfo(CL_IMAGE_HEIGHT, &height);
    src.getImageInfo(CL_IMAGE_FORMAT, &format);

    ::cl::Image2D scratch(context, CL_MEM_READ_WRITE, ::cl::ImageFormat(format.image_channel_order, format.image_channel_data_type), width, height);

    separableConvolve(context, commandQueue, src, width, height, kernelBuffer, kernelSize, kernelBuffer, kernelSize, 1.0, dst, scratch, events, event);
}

::cl::Buffer buildScharrFilter(::cl::Context context, int scale)
{
    ::cl::Buffer kernel;
    const int kernelSize=3+2*(scale-1);

//    std::vector<float> kernel
    return kernel;
}

ScharrSeparableKernel buildScharrSeparableKernel(::cl::Context context, int size, int &kernelSize, bool normalize)
{
    ScharrSeparableKernel kernel;

    kernelSize=(2*size)+1;
    
    if(false)
    {
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> smoothingKernel;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> edgeKernel;
        Eigen::Vector3f smoothingVector;
        Eigen::Vector3f edgeVector;

        float smoothNorm=16.0;
        float edgeNorm=2.0;

        smoothingVector<<3, 10, 3;
        edgeVector<<-1, 0, 1;

        smoothingKernel=smoothingVector;
        edgeKernel=edgeVector;

        while(size>1)
        {
            smoothingKernel=convolveMatrix(smoothingKernel, smoothingVector);
            smoothNorm*=16.0;
            edgeKernel=convolveMatrix(edgeKernel, smoothingVector);
            edgeNorm*=16.0;
            size--;
        }

        if(normalize)
        {
            smoothingKernel/=smoothNorm;
            edgeKernel/=edgeNorm;
        }

        kernel.smooth=::cl::Buffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*smoothingKernel.size(), smoothingKernel.data());
        kernel.edge=::cl::Buffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*edgeKernel.size(), edgeKernel.data());
    }
    else
    {
        Eigen::RowVectorXf smoothingVector(kernelSize);
        Eigen::RowVectorXf edgeVector(kernelSize);

        smoothingVector.setZero();
        if(normalize)
        {
            float w=10.0/3.0;
            float norm=1.0/(2.0 * size * (w+2.0));

            smoothingVector(0)=norm;
            smoothingVector(kernelSize/2)=w * norm;
            smoothingVector(kernelSize-1)=norm;
        }
        else
        {
            smoothingVector(0)=3;
            smoothingVector(kernelSize/2)=10;
            smoothingVector(kernelSize-1)=3;
        }

        edgeVector.setZero();
        edgeVector(0)=-1.0;
        edgeVector(kernelSize-1)=1.0;

        kernel.smooth=::cl::Buffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*smoothingVector.size(), smoothingVector.data());
        kernel.edge=::cl::Buffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*edgeVector.size(), edgeVector.data());
    }
    return kernel;
}

//void scharrConvolution(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, ::cl::Image2D &dst, size_t width, size_t height, float sigma, std::vector<::cl::Event> *events, ::cl::Event event)
//{
//    size_t filterSize;
//
//    builsdScharrKernel(sigma, ksize_x, ksize_y);
//    ::cl::Buffer filterBuffer=buildGaussianFilter(context, sigma, filterSize);
//
//    ::cl::Kernel kernel=getKernel(context, "convolve", "kernels/convolve.cl");
//    cl_int status;
//    int index=0;
//
//    status=kernel.setArg(index++, src);
//    status=kernel.setArg(index++, width);
//    status=kernel.setArg(index++, filterBuffer);
//    status=kernel.setArg(index++, filterSize);
//    status=kernel.setArg(index++, dst);
//
//    ::cl::NDRange globalThreads(width, height);
//
//    status=commandQueue.enqueueNDRangeKernel(kernel, ::cl::NullRange, globalThreads, ::cl::NullRange, events, &event);
//}

void scharrSeparable(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, size_t width, size_t height, int size, float scale, bool yKernel, bool normalize, ::cl::Image2D &dst, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    int kernelSize;
    ScharrSeparableKernel kernelBuffer=buildScharrSeparableKernel(context, size, kernelSize, normalize);
    cl_image_format format;

    src.getImageInfo(CL_IMAGE_FORMAT, &format);

    ::cl::Image2D scratch(context, CL_MEM_READ_WRITE, ::cl::ImageFormat(format.image_channel_order, format.image_channel_data_type), width, height);

    if(yKernel)
        separableConvolve(context, commandQueue, src, width, height, kernelBuffer.smooth, kernelSize, kernelBuffer.edge, kernelSize, scale, dst, scratch, events, event);
    else
        separableConvolve(context, commandQueue, src, width, height, kernelBuffer.edge, kernelSize, kernelBuffer.smooth, kernelSize, scale, dst, scratch, events, event);
}

void scharrSeparable(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, size_t width, size_t height, int size, float scale, bool yKernel, bool normalize, ::cl::Buffer &dst, size_t offset, 
    std::vector<::cl::Event> *events, ::cl::Event &event)
{
    int kernelSize;
    ScharrSeparableKernel kernelBuffer=buildScharrSeparableKernel(context, size, kernelSize, normalize);
    cl_image_format format;

    src.getImageInfo(CL_IMAGE_FORMAT, &format);

    ::cl::Image2D scratch(context, CL_MEM_READ_WRITE, ::cl::ImageFormat(format.image_channel_order, format.image_channel_data_type), width, height);

    if(yKernel)
        separableConvolve(context, commandQueue, src, width, height, kernelBuffer.smooth, kernelSize, kernelBuffer.edge, kernelSize, scale, dst, offset, scratch, events, event);
    else
        separableConvolve(context, commandQueue, src, width, height, kernelBuffer.edge, kernelSize, kernelBuffer.smooth, kernelSize, scale, dst, offset, scratch, events, event);
}

void scharrSeparable(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Buffer &src, size_t srcOffset, size_t width, size_t height, int size, float scale, bool yKernel, bool normalize, 
    ::cl::Buffer &dst, size_t dstOffset, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    int kernelSize;
    ScharrSeparableKernel kernelBuffer=buildScharrSeparableKernel(context, size, kernelSize, normalize);
    cl_image_format format;

    ::cl::Buffer scratch(context, CL_MEM_READ_WRITE, width*height*sizeof(cl_float));

    if(yKernel)
        separableConvolve(context, commandQueue, src, srcOffset, width, height, kernelBuffer.smooth, kernelSize, kernelBuffer.edge, kernelSize, scale, dst, dstOffset, scratch, events, event);
    else
        separableConvolve(context, commandQueue, src, srcOffset, width, height, kernelBuffer.edge, kernelSize, kernelBuffer.smooth, kernelSize, scale, dst, dstOffset, scratch, events, event);
}

//double lowPass(CudaImage &inimg, CudaImage &outimg, CudaImage &temp, double var, int kernsize)
//{
//    cl::Kernel kernel=getMapKernel(m_type);
//
//    size_t mapSize=mapValueSize*m_sourceCount*m_outputWidth*m_outputHeight;
//
//    if(mapSize > m_mapBufferSize)
//    {
//        cl_int error=CL_SUCCESS;
//
//        m_mapBuffer=cl::Buffer(m_openCLContext, CL_MEM_READ_WRITE, mapSize*sizeof(float), NULL, &error);
//        m_mapBufferSize=mapSize;
//    }
//
//    status=kernel.setArg(index++, m_mappingBuffer);
//    status=kernel.setArg(index++, m_sourceCount);
//    status=kernel.setArg(index++, m_parametersBuffer);
//    status=kernel.setArg(index++, m_mapBuffer);
//
//    cl::NDRange globalThreads(m_outputWidth, m_outputHeight);
//
//    status=m_openCLComandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, globalThreads, cl::NullRange, NULL, &event);
//
//
//    if(kernsize<=5)
//        return lowPass_2(inimg, outimg, temp, var);
//    else if(kernsize<=7)
//        return lowPass_3(inimg, outimg, temp, var);
//    else if(kernsize<=9)
//        return lowPass_4(inimg, outimg, temp, var);
//    else
//    {
//        if(kernsize > 11)
//            std::cerr<<"Kernels larger than 11 not implemented"<<std::endl;
//        return lowPass_5(inimg, outimg, temp, var);
//    }
//}

void calculateMagnitude(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &lx, ::cl::Image2D &ly, ::cl::Image2D &magnitude, size_t width, size_t height, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    ::cl::Kernel kernel=getKernel(context, "magnitude", "lib/kernels/convolve.cl");
    cl_int status;
    int index=0;

    status=kernel.setArg(index++, lx);
    status=kernel.setArg(index++, ly);
    status=kernel.setArg(index++, magnitude);

    ::cl::NDRange globalThreads(width, height);

    status=commandQueue.enqueueNDRangeKernel(kernel, ::cl::NullRange, globalThreads, ::cl::NullRange, events, &event);

    commandQueue.flush();
}

float calculateMax(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, size_t width, size_t height, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    std::vector<float> maxVector(height);
    ::cl::Buffer maxBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float)*maxVector.size());

    ::cl::Kernel kernel=getKernel(context, "rowMax", "lib/kernels/convolve.cl");
    ::cl::Event rowMaxEvent;
    cl_int status;
    int index=0;

    status=kernel.setArg(index++, src);
    status=kernel.setArg(index++, (int)width);
    status=kernel.setArg(index++, maxBuffer);

    ::cl::NDRange globalThreads(height, 1);

    status=commandQueue.enqueueNDRangeKernel(kernel, ::cl::NullRange, globalThreads, ::cl::NullRange, events, &rowMaxEvent);

    std::vector<::cl::Event> rowMaxCompleteEvent={rowMaxEvent};

    commandQueue.enqueueReadBuffer(maxBuffer, false, 0, sizeof(cl_float)*maxVector.size(), maxVector.data(), &rowMaxCompleteEvent, &event);

    commandQueue.flush();
    event.wait();

    float maxValue=maxVector[0];
    for(size_t i=1; i<height; ++i)
    { 
        if(maxVector[i]>maxValue)
            maxValue=maxVector[i];
    }

    return maxValue;
}

std::vector<int> calculateHistogram(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, size_t width, size_t height, int bins, float scale, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    std::vector<int> histogram(bins);
    ::cl::Buffer histogramBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int)*histogram.size());// , histogram.data());
    ::cl::Buffer scratchBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int)*height*bins);

    ::cl::Kernel kernelHistogramRows=getKernel(context, "histogramRows", "lib/kernels/convolve.cl");
    ::cl::Kernel kernelHistogramCombine=getKernel(context, "histogramCombine", "lib/kernels/convolve.cl");

    ::cl::Event histogramRowEvent;
    ::cl::Event histogramCombineEvent;
    cl_int status;
    int index=0;
    int items=32;

    status=kernelHistogramRows.setArg(index++, src);
    status=kernelHistogramRows.setArg(index++, (int)width);
    status=kernelHistogramRows.setArg(index++, (int)height);
    status=kernelHistogramRows.setArg(index++, bins);
    status=kernelHistogramRows.setArg(index++, scale);
    status=kernelHistogramRows.setArg(index++, scratchBuffer);
//    status=kernelHistogramRows.setArg(index++, items*bins, nullptr);
//
//    int workGroups=ceil((float)height/items);
//    int workItems=items*workGroups;
//
//    ::cl::NDRange globalThreads(workItems);
//    ::cl::NDRange localThreads(items);
//
//    status=commandQueue.enqueueNDRangeKernel(kernelHistogramRows, ::cl::NullRange, globalThreads, localThreads, events, &histogramRowEvent);
    ::cl::NDRange globalThreads(height);

    status=commandQueue.enqueueNDRangeKernel(kernelHistogramRows, ::cl::NullRange, globalThreads, ::cl::NullRange, events, &histogramRowEvent);

    std::vector<::cl::Event> histogramRowCompleteEvent={histogramRowEvent};
    
//    {
//        std::vector<int> scratch(height*bins);
//        ::cl::Event scratchReadEvent;
//
//        status=commandQueue.enqueueReadBuffer(scratchBuffer, false, 0, sizeof(cl_int)*scratch.size(), scratch.data(), &histogramRowCompleteEvent, &scratchReadEvent);
//
//        scratchReadEvent.wait();
//        scratchReadEvent.wait();
//    }
   
    index=0;
    status=kernelHistogramCombine.setArg(index++, scratchBuffer);
    status=kernelHistogramCombine.setArg(index++, bins);
    status=kernelHistogramCombine.setArg(index++, (int)height);
    status=kernelHistogramCombine.setArg(index++, histogramBuffer);
//    __kernel void histogramCombine(__constant int *input, int bins, int count, __global int *output)

    ::cl::NDRange globalBinThreads(bins);

    status=commandQueue.enqueueNDRangeKernel(kernelHistogramCombine, ::cl::NullRange, globalBinThreads, ::cl::NullRange, &histogramRowCompleteEvent, &histogramCombineEvent);

    std::vector<::cl::Event> histogramCombineCompleteEvent={histogramCombineEvent};

    status=commandQueue.enqueueReadBuffer(histogramBuffer, false, 0, sizeof(cl_int)*histogram.size(), histogram.data(), &histogramCombineCompleteEvent, &event);

    commandQueue.flush();
    event.wait();

    return histogram;
}

float computeKPercentile(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, float perc, float gscale, size_t nbins, size_t ksize_x, size_t ksize_y)
{
    size_t nbin=0, nelements=0, nthreshold=0, k=0;
    float kperc=0.0, modg=0.0, npoints=0.0, hmax=0.0;

    size_t width, height;
    cl_image_format format;

    src.getImageInfo(CL_IMAGE_WIDTH, &width);
    src.getImageInfo(CL_IMAGE_HEIGHT, &height);
    src.getImageInfo(CL_IMAGE_FORMAT, &format);

    ::cl::Image2D gaussian(context, CL_MEM_READ_WRITE, ::cl::ImageFormat(format.image_channel_order, format.image_channel_data_type), width, height);
    ::cl::Image2D lx(context, CL_MEM_READ_WRITE, ::cl::ImageFormat(format.image_channel_order, format.image_channel_data_type), width, height);
    ::cl::Image2D ly(context, CL_MEM_READ_WRITE, ::cl::ImageFormat(format.image_channel_order, format.image_channel_data_type), width, height);
    ::cl::Image2D magnitude(context, CL_MEM_READ_WRITE, ::cl::ImageFormat(format.image_channel_order, format.image_channel_data_type), width, height);

    std::vector<float> histogram(nbins);
    ::cl::Buffer histogramBuffer(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, sizeof(cl_float)*histogram.size(), histogram.data());

    ::cl::Event guassianEvent;
    
    // Perform the Gaussian convolution
    gaussianSeparable(context, commandQueue, src, gaussian, width, height, gscale, nullptr, guassianEvent);

    std::vector<::cl::Event> waitEvent={guassianEvent};
    std::vector<::cl::Event> derivativeEvent(2);

    // Compute the Gaussian derivatives Lx and Ly
    scharrSeparable(context, commandQueue, gaussian, width, height, 1, 1.0, false, false, lx, &waitEvent, derivativeEvent[0]);
    scharrSeparable(context, commandQueue, gaussian, width, height, 1, 1.0, true, false, ly, &waitEvent, derivativeEvent[1]);

    // Calculate magnitude
    ::cl::Event combinedEvent;
    calculateMagnitude(context, commandQueue, lx, ly, magnitude, width, height, &derivativeEvent, combinedEvent);

    // Get the maximum from the magnitude
    std::vector<::cl::Event> combinedWaitEvent={combinedEvent};
    ::cl::Event maxEvent;
    hmax=calculateMax(context, commandQueue, magnitude, width, height, &combinedWaitEvent, maxEvent);

    ::cl::Event histogramEvent;

    //Create scaled histogram using nbins and max of magnitude
    float maxScale=1/hmax;
    std::vector<int> hist=calculateHistogram(context, commandQueue, magnitude, width, height, nbins, maxScale, nullptr, histogramEvent);

    for(size_t i=0; i<hist.size(); ++i)
        npoints+=hist[i];

    // Now find the perc of the histogram percentile
    nthreshold=(size_t)(npoints * perc);

    for(k=0; nelements < nthreshold && k < nbins; k++)
        nelements=nelements+hist[k];

    if(nelements < nthreshold)
        kperc=0.03;
    else
        kperc=hmax * ((float)(k)/(float)nbins);

    return kperc;
}

void linearSample(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, ::cl::Image2D &dst, size_t dstWidth, size_t dstHeight, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    ::cl::Kernel kernel=getKernel(context, "linearSample", "lib/kernels/convolve.cl");
    cl_int status;
    int index=0;

    status=kernel.setArg(index++, src);
    status=kernel.setArg(index++, dst);
    status=kernel.setArg(index++, (int)dstWidth);
    status=kernel.setArg(index++, (int)dstHeight);

    ::cl::NDRange globalThreads(dstWidth, dstHeight);

    status=commandQueue.enqueueNDRangeKernel(kernel, ::cl::NullRange, globalThreads, ::cl::NullRange, events, &event);

    commandQueue.flush();
}

void pmG1(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src1, ::cl::Image2D &src2, ::cl::Image2D &dst, size_t width, size_t height, float contrast, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    ::cl::Kernel kernel=getKernel(context, "pmG1", "lib/kernels/convolve.cl");
    cl_int status;
    int index=0;

    status=kernel.setArg(index++, src1);
    status=kernel.setArg(index++, src2);
    status=kernel.setArg(index++, dst);
    status=kernel.setArg(index++, contrast);

    ::cl::NDRange globalThreads(width, height);

    status=commandQueue.enqueueNDRangeKernel(kernel, ::cl::NullRange, globalThreads, ::cl::NullRange, events, &event);

    commandQueue.flush();
}

void pmG2(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src1, ::cl::Image2D &src2, ::cl::Image2D &dst, size_t width, size_t height, float contrast, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    ::cl::Kernel kernel=getKernel(context, "pmG2", "lib/kernels/convolve.cl");
    cl_int status;
    int index=0;

    status=kernel.setArg(index++, src1);
    status=kernel.setArg(index++, src2);
    status=kernel.setArg(index++, dst);
    status=kernel.setArg(index++, contrast);

    ::cl::NDRange globalThreads(width, height);

    status=commandQueue.enqueueNDRangeKernel(kernel, ::cl::NullRange, globalThreads, ::cl::NullRange, events, &event);

    commandQueue.flush();
}

void weickert(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src1, ::cl::Image2D &src2, ::cl::Image2D &dst, size_t width, size_t height, float contrast, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    ::cl::Kernel kernel=getKernel(context, "weickert", "lib/kernels/convolve.cl");
    cl_int status;
    int index=0;

    status=kernel.setArg(index++, src1);
    status=kernel.setArg(index++, src2);
    status=kernel.setArg(index++, dst);
    status=kernel.setArg(index++, contrast);

    ::cl::NDRange globalThreads(width, height);

    status=commandQueue.enqueueNDRangeKernel(kernel, ::cl::NullRange, globalThreads, ::cl::NullRange, events, &event);

    commandQueue.flush();
}

void charbonnier(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src1, ::cl::Image2D &src2, ::cl::Image2D &dst, size_t width, size_t height, float contrast, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    ::cl::Kernel kernel=getKernel(context, "charbonnier", "lib/kernels/convolve.cl");
    cl_int status;
    int index=0;

    status=kernel.setArg(index++, src1);
    status=kernel.setArg(index++, src2);
    status=kernel.setArg(index++, dst);
    status=kernel.setArg(index++, contrast);

    ::cl::NDRange globalThreads(width, height);

    status=commandQueue.enqueueNDRangeKernel(kernel, ::cl::NullRange, globalThreads, ::cl::NullRange, events, &event);

    commandQueue.flush();
}

void nldStepScalar(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &image, ::cl::Image2D &flow, ::cl::Image2D &step, size_t width, size_t height, float stepsize, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    ::cl::Kernel kernel=getKernel(context, "nldStepScalar", "lib/kernels/convolve.cl");
    cl_int status;
    int index=0;

    status=kernel.setArg(index++, image);
    status=kernel.setArg(index++, flow);
    status=kernel.setArg(index++, step);
    status=kernel.setArg(index++, stepsize);

    ::cl::NDRange globalThreads(width, height);

    status=commandQueue.enqueueNDRangeKernel(kernel, ::cl::NullRange, globalThreads, ::cl::NullRange, events, &event);

    commandQueue.flush();
}

void scale(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &image, size_t width, size_t height, float scale, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    ::cl::Kernel kernel=getKernel(context, "scale", "lib/kernels/convolve.cl");
    cl_int status;
    int index=0;
    size_t workGroupSizeMultiple;
    int items=width*height;

    ::cl::Device device=context.getInfo<CL_CONTEXT_DEVICES>()[0];
    kernel.getWorkGroupInfo(device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &workGroupSizeMultiple);

    status=kernel.setArg(index++, image);
    status=kernel.setArg(index++, scale);
    status=kernel.setArg(index++, items);

    int workItems=ceil((float)items/workGroupSizeMultiple)*workGroupSizeMultiple;

    ::cl::NDRange globalThreads(workItems);
    ::cl::NDRange localThreads(workGroupSizeMultiple);

    status=commandQueue.enqueueNDRangeKernel(kernel, ::cl::NullRange, globalThreads, localThreads, events, &event);

    commandQueue.flush();
}

void determinantHessian(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &dxx, ::cl::Image2D &dyy, ::cl::Image2D &dxy, size_t width, size_t height, ::cl::Buffer &output, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    ::cl::Kernel kernel=getKernel(context, "determinantHessian", "lib/kernels/convolve.cl");
    cl_int status;
    int index=0;

    status=kernel.setArg(index++, dxx);
    status=kernel.setArg(index++, dyy);
    status=kernel.setArg(index++, dxy);
    status=kernel.setArg(index++, output);

    ::cl::NDRange globalThreads(width, height);
    
    status=commandQueue.enqueueNDRangeKernel(kernel, ::cl::NullRange, globalThreads, ::cl::NullRange, events, &event);

    commandQueue.flush();
}

void determinantHessian(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Buffer &dxx, size_t dxxOffset, ::cl::Buffer &dyy, size_t dyyOffset, ::cl::Buffer &dxy, size_t dxyOffset, 
    size_t width, size_t height, ::cl::Buffer &output, size_t outputOffset, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    ::cl::Kernel kernel=getKernel(context, "determinantHessianBuffer", "lib/kernels/convolve.cl");
    cl_int status;
    int index=0;

    status=kernel.setArg(index++, dxx);
    status=kernel.setArg(index++, (int)dxxOffset);
    status=kernel.setArg(index++, dyy);
    status=kernel.setArg(index++, (int)dyyOffset);
    status=kernel.setArg(index++, dxy);
    status=kernel.setArg(index++, (int)dxyOffset);
    status=kernel.setArg(index++, (int)width);
    status=kernel.setArg(index++, (int)height);
    status=kernel.setArg(index++, output);
    status=kernel.setArg(index++, (int)outputOffset);

    ::cl::NDRange globalThreads(width, height);

    status=commandQueue.enqueueNDRangeKernel(kernel, ::cl::NullRange, globalThreads, ::cl::NullRange, events, &event);

    commandQueue.flush();
}

void findExtrema(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &detImage, size_t width, size_t height, ::cl::Buffer &featureMap, size_t featureMapWidth, size_t featureMapHeight,
    int evolutionClass, int octave, float sigma, float threshold, float derivativeFactor, int border, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    ::cl::Kernel kernel=getKernel(context, "findExtrema", "lib/kernels/convolve.cl");
    cl_int status;
    int index=0;

    status=kernel.setArg(index++, detImage);
    status=kernel.setArg(index++, (int)width);
    status=kernel.setArg(index++, (int)height);
    status=kernel.setArg(index++, featureMap);
    status=kernel.setArg(index++, (int)featureMapWidth);
    status=kernel.setArg(index++, (int)featureMapHeight);
    status=kernel.setArg(index++, evolutionClass);
    status=kernel.setArg(index++, octave);
    status=kernel.setArg(index++, sigma);
    status=kernel.setArg(index++, threshold);
    status=kernel.setArg(index++, derivativeFactor);
    status=kernel.setArg(index++, border);

    ::cl::NDRange globalThreads(width, height);

    status=commandQueue.enqueueNDRangeKernel(kernel, ::cl::NullRange, globalThreads, ::cl::NullRange, events, &event);

    commandQueue.flush();
}

void findExtremaIterate(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &detImage, size_t width, size_t height, ::cl::Buffer &featureMap, size_t featureMapWidth, size_t featureMapHeight,
    int evolutionClass, int octave, float sigma, float threshold, float derivativeFactor, int border, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    ::cl::Kernel kernel=getKernel(context, "findExtremaIterate", "lib/kernels/convolve.cl");
    cl_int status;
    int index=0;

    status=kernel.setArg(index++, detImage);
    status=kernel.setArg(index++, (int)width);
    status=kernel.setArg(index++, (int)height);
    status=kernel.setArg(index++, featureMap);
    status=kernel.setArg(index++, (int)featureMapWidth);
    status=kernel.setArg(index++, (int)featureMapHeight);
    status=kernel.setArg(index++, evolutionClass);
    status=kernel.setArg(index++, octave);
    status=kernel.setArg(index++, sigma);
    status=kernel.setArg(index++, threshold);
    status=kernel.setArg(index++, derivativeFactor);
    status=kernel.setArg(index++, border);

    ::cl::NDRange globalThreads(width-(2*border), height-(2*border));

    status=commandQueue.enqueueNDRangeKernel(kernel, ::cl::NullRange, globalThreads, ::cl::NullRange, events, &event);

    commandQueue.flush();
}

void findExtrema(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Buffer &detImage, ::cl::Buffer &featureMap, int width, int height, int mapIndex, ::cl::Buffer evolutionBuffer, float threshold, float derivativeFactor, int border,
	::cl::Buffer keypointCount, std::vector<::cl::Event> *events, ::cl::Event &event)
{
	::cl::Kernel kernel = getKernel(context, "findExtremaBuffer", "lib/kernels/convolve.cl");
	cl_int status;
	int index = 0;

	//__kernel void findExtremaBuffer(__global float *input, __global ExtremaMap *extremaMap, __global EvolutionInfo *evolutionBuffer, int index, float threshold, float derivativeFactor, __global int *keypointCount)
//    size_t workGroupSizeMultiple;
//    
//    ::cl::Device device=context.getInfo<CL_CONTEXT_DEVICES>()[0];
//    kernel.getWorkGroupInfo(device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &workGroupSizeMultiple);

    int workX=width-(2*border);
    int workY=height-(2*border);

    if((workX<=0)||(workY<=0))
        return;

//    int workXItems=ceil((float)workX/workGroupSizeMultiple)*workGroupSizeMultiple;
//    int workYItems=ceil((float)workY/workGroupSizeMultiple)*workGroupSizeMultiple;
	
    status=kernel.setArg(index++, detImage);
	status=kernel.setArg(index++, featureMap);
	status=kernel.setArg(index++, mapIndex);
	status=kernel.setArg(index++, evolutionBuffer);
	status=kernel.setArg(index++, threshold);
	status=kernel.setArg(index++, border);
	status=kernel.setArg(index++, keypointCount);
    status=kernel.setArg(index++, (int)workX);
    status=kernel.setArg(index++, (int)workY);

//	::cl::NDRange globalThreads(workXItems, workYItems);
//	::cl::NDRange localThreads(workGroupSizeMultiple, workGroupSizeMultiple);
//
//	status = commandQueue.enqueueNDRangeKernel(kernel, ::cl::NullRange, globalThreads, localThreads, events, &event);

    ::cl::NDRange globalThreads(workX, workY);

    status=commandQueue.enqueueNDRangeKernel(kernel, ::cl::NullRange, globalThreads, ::cl::NullRange, events, &event);

	commandQueue.flush();
}

void consolidateKeypoints(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Buffer &featureMap, int width, int height, int mapIndex, ::cl::Buffer evolutionBuffer, int border,
    ::cl::Buffer keypoints, int maxKeypoints, ::cl::Buffer count, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    ::cl::Kernel kernel=getKernel(context, "consolidateKeypoints", "lib/kernels/convolve.cl");
    cl_int status;
    int index=0;

    int workX=width-(2*border);
    int workY=height-(2*border);

    if((workX<=0)||(workY<=0))
        return;
    //__kernel void consolidateKeypoints(__global ExtremaMap *extremaMapBuffer, int index, __global EvolutionInfo *evolutionBuffer, int workOffset, __global Keypoint *keypoints, int maxKeypoints, __global int *count)

    status=kernel.setArg(index++, featureMap);
    status=kernel.setArg(index++, mapIndex);
    status=kernel.setArg(index++, evolutionBuffer);
    status=kernel.setArg(index++, border);
    status=kernel.setArg(index++, keypoints);
    status=kernel.setArg(index++, maxKeypoints);
    status=kernel.setArg(index++, count);

    ::cl::NDRange globalThreads(workX, workY);

    status=commandQueue.enqueueNDRangeKernel(kernel, ::cl::NullRange, globalThreads, ::cl::NullRange, events, &event);

    commandQueue.flush();
}

void subPixelRefinement(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Buffer &det, ::cl::Buffer evolutionBuffer, ::cl::Buffer keypoints, int keypointCount, 
    ::cl::Buffer filteredKeypoints, ::cl::Buffer keypointCountBuffer, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    ::cl::Kernel kernel=getKernel(context, "subPixelRefinement", "lib/kernels/convolve.cl");
    cl_int status;
    int index=0;

    status=kernel.setArg(index++, det);
    status=kernel.setArg(index++, evolutionBuffer);
    status=kernel.setArg(index++, keypoints);
    status=kernel.setArg(index++, filteredKeypoints);
    status=kernel.setArg(index++, keypointCountBuffer);

    ::cl::NDRange globalThreads(keypointCount, 1);

    status=commandQueue.enqueueNDRangeKernel(kernel, ::cl::NullRange, globalThreads, ::cl::NullRange, events, &event);

    commandQueue.flush();
}

void computeOrientation(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Buffer &keypoints, size_t count, ::cl::Buffer &dx, ::cl::Buffer &dy,
    ::cl::Buffer &evolutionInfo, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    ::cl::Kernel kernel=getKernel(context, "computeOrientation", "lib/kernels/convolve.cl");
    cl_int status;
    int index=0;

    size_t dataSize=((109*3)+(42*3))*count;
    ::cl::Buffer dataBuffer=::cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*dataSize);

    status=kernel.setArg(index++, keypoints);
    status=kernel.setArg(index++, dx);
    status=kernel.setArg(index++, dy);
    status=kernel.setArg(index++, evolutionInfo);
    status=kernel.setArg(index++, dataBuffer);
//    status=kernel.setArg(index++, (109*3)+1, nullptr); //allocate local memory: 3x 109 points + 1 for max value calculation

    ::cl::NDRange globalThreads(count, 42);
    ::cl::NDRange localThreads(1, 42);

    status=commandQueue.enqueueNDRangeKernel(kernel, ::cl::NullRange, globalThreads, localThreads, events, &event);

//    event.wait();
//
//    std::vector<float> data(dataSize);
//    std::vector<cl::Keypoint> keypointVector(count);
//    
//    commandQueue.enqueueReadBuffer(keypoints, CL_FALSE, 0, keypointVector.size()*sizeof(cl::Keypoint), keypointVector.data(), nullptr, &event);
//    event.wait();
//    commandQueue.enqueueReadBuffer(dataBuffer, CL_FALSE, 0, data.size()*sizeof(float), data.data(), nullptr, &event);
//    event.wait();
//
//    std::ofstream file("values_cl.txt");
//
//    for(size_t i=0; i<count; ++i)
//    {
//        size_t dataOffset=((109*3)+(42*3))*i;
//
//        file<<"Keypoint: "<<i<<", "<<keypointVector[i].ptX<<", "<<keypointVector[i].ptY;
//        file<<"\n Values \n";
//        for(size_t j=0; j<109; ++j)
//        {
//            file<<"("<<data[dataOffset+(j*3)]<<", "<<data[dataOffset+(j*3)+1]<<") "<<data[dataOffset+(j*3)+2]<<"\n";
//        }
//        file<<"\nAngles \n";
//
//        dataOffset+=(109*3);
//        for(size_t j=0; j<42; ++j)
//        {
//            file<<"("<<data[dataOffset+(j*3)]<<", "<<data[dataOffset+(j*3)+1]<<") "<<data[dataOffset+(j*3)+2]<<"\n";
//        }
//    }

    commandQueue.flush();
}

void getMLDBDescriptors(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Buffer &keypoints, ::cl::Buffer descriptors, size_t count, ::cl::Buffer &image, ::cl::Buffer &dx, ::cl::Buffer &dy,
    ::cl::Buffer &evolutionInfo, int channels, int patternSize, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    ::cl::Kernel kernel=getKernel(context, "getMLDBDescriptor", "lib/kernels/convolve.cl");
    cl_int status;
    int index=0;

    //__kernel void getMLDBDescriptor(__global Keypoint *keypoints, __global unsigned char *desc, __global float *imageBuffer, __global float *dxBuffer, __global float *dyBuffer, __global EvolutionInfo *evolution, int patternSize)

    status=kernel.setArg(index++, keypoints);
    status=kernel.setArg(index++, descriptors);
    status=kernel.setArg(index++, image);
    status=kernel.setArg(index++, dx);
    status=kernel.setArg(index++, dy);
    status=kernel.setArg(index++, evolutionInfo);
//    status=kernel.setArg(index++, channels);
    status=kernel.setArg(index++, patternSize);
    
    ::cl::NDRange globalThreads(count);

    status=commandQueue.enqueueNDRangeKernel(kernel, ::cl::NullRange, globalThreads, ::cl::NullRange, events, &event);

    commandQueue.flush();
}

}}//namespace libAKAZE::cl