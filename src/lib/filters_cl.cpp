#include "filters_cl.h"
#include "nldiffusion_functions.h"

#include "openClContext.h"

#include "AKAZE_cl.h"
#include <fstream>

namespace libAKAZE
{
namespace cl
{

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
}

float calculateMax(::cl::Context &context, ::cl::CommandQueue &commandQueue, ::cl::Image2D &src, size_t width, size_t height, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    float maxValue;
    ::cl::Buffer maxBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float));
    ::cl::Event rowMaxEvent;

    calculateMax(context, commandQueue, src, width, height, maxBuffer, events, rowMaxEvent);
    std::vector<::cl::Event> rowMaxCompleteEvent={rowMaxEvent};

    commandQueue.enqueueReadBuffer(maxBuffer, false, 0, sizeof(cl_float), &maxValue, &rowMaxCompleteEvent, &event);

    commandQueue.flush();
    event.wait();

    return maxValue;
}

void calculateMax(::cl::Context &context, ::cl::CommandQueue &commandQueue, ::cl::Image2D &src, size_t width, size_t height, ::cl::Buffer &maxBuffer, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    ::cl::Kernel kernel=getKernel(context, "rowMax", "lib/kernels/convolve.cl");
    ::cl::Event rowMaxEvent;
    cl_int status;
    int index=0;

    status=kernel.setArg(index++, src);
    status=kernel.setArg(index++, (int)width);
    status=kernel.setArg(index++, maxBuffer);

    ::cl::NDRange globalThreads(height, 1);

    status=commandQueue.enqueueNDRangeKernel(kernel, ::cl::NullRange, globalThreads, ::cl::NullRange, events, &event);
}

std::vector<int> calculateHistogram(::cl::Context &context, ::cl::CommandQueue &commandQueue, ::cl::Image2D &src, size_t width, size_t height, int bins, float maxValue, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    std::vector<int> histogram(bins);
    
    ::cl::Buffer histogramBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int)*histogram.size());// , histogram.data());
    ::cl::Buffer scratchBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int)*height*bins);
    ::cl::Buffer maxValueBuffer(context, CL_MEM_READ_WRITE, sizeof(float));

    std::vector<::cl::Event> writeMaxValueEvent(1);
    std::vector<::cl::Event> histogramCombineCompleteEvent(1);

    commandQueue.enqueueWriteBuffer(maxValueBuffer, false, 0, sizeof(float), &scale, events, &writeMaxValueEvent[0]);

    calculateHistogram(context, commandQueue, src, width, height, bins, maxValueBuffer,
        histogramBuffer, scratchBuffer, &writeMaxValueEvent, histogramCombineCompleteEvent[0]);

    commandQueue.enqueueReadBuffer(histogramBuffer, false, 0, sizeof(cl_int)*histogram.size(), histogram.data(), &histogramCombineCompleteEvent, &event);

    commandQueue.flush();
    event.wait();

    return histogram;
}

void calculateHistogram(::cl::Context &context, ::cl::CommandQueue &commandQueue, ::cl::Image2D &src, size_t width, size_t height, int bins, ::cl::Buffer maxValueBuffer, 
    ::cl::Buffer &histogramBuffer, ::cl::Buffer &scratchBuffer, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    ::cl::Kernel kernelHistogramRows=getKernel(context, "histogramRows", "lib/kernels/convolve.cl");
    ::cl::Kernel kernelHistogramCombine=getKernel(context, "histogramCombine", "lib/kernels/convolve.cl");

    ::cl::Event histogramRowEvent;
    cl_int status;
    int index=0;
    int items=32;

    status=kernelHistogramRows.setArg(index++, src);
    status=kernelHistogramRows.setArg(index++, (int)width);
    status=kernelHistogramRows.setArg(index++, (int)height);
    status=kernelHistogramRows.setArg(index++, bins);
    status=kernelHistogramRows.setArg(index++, maxValueBuffer);
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

    status=commandQueue.enqueueNDRangeKernel(kernelHistogramCombine, ::cl::NullRange, globalBinThreads, ::cl::NullRange, &histogramRowCompleteEvent, &event);

    
}

void computeContrast(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, size_t width, size_t height, ::cl::Image2D &dst, std::vector<::cl::Event> *events, ::cl::Event &event)
{
//    ::cl::Kernel kernel=getKernel(context, "computeContrast", "lib/kernels/contrast.cl");
    SharedKernelInfo kernelInfo=getKernelInfo(context, "computeContrast", "lib/kernels/contrast.cl");
    ::cl::Kernel kernel=kernelInfo->kernel;

    cl_int status;
    int index=0;

    size_t localX=16;
    size_t localY=16;
    size_t globalX=(width/localX)*localX;
    size_t globalY=(height/localY)*localY;

    if(globalX<width)
        globalX+=localX;
    if(globalY<height)
        globalY+=localY;

    status=kernel.setArg(index++, src);
    status=kernel.setArg(index++, (int)width);
    status=kernel.setArg(index++, (int)height);
    status=kernel.setArg(index++, dst);
//    status=kernel.setArg(index++, imageCache, nullptr);
//    status=kernel.setArg(index++, imageCache);

    ::cl::NDRange globalThreads(globalX, globalY);
    ::cl::NDRange localThreads(localX, localY);

    status=commandQueue.enqueueNDRangeKernel(kernel, ::cl::NullRange, globalThreads, localThreads, events, &event);
}

float computeKPercentile(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, float perc, float gscale, size_t nbins)
{
    size_t width, height;
    cl_image_format format;

    src.getImageInfo(CL_IMAGE_WIDTH, &width);
    src.getImageInfo(CL_IMAGE_HEIGHT, &height);
    src.getImageInfo(CL_IMAGE_FORMAT, &format);

    ::cl::Image2D gaussian(context, CL_MEM_READ_WRITE, ::cl::ImageFormat(format.image_channel_order, format.image_channel_data_type), width, height);
    ::cl::Image2D magnitude(context, CL_MEM_READ_WRITE, ::cl::ImageFormat(format.image_channel_order, format.image_channel_data_type), width, height);
    ::cl::Image2D scratch(context, CL_MEM_READ_WRITE, ::cl::ImageFormat(format.image_channel_order, format.image_channel_data_type), width, height);

    int gussianKernelSize;
    int scharrKernelSize;

    ::cl::Buffer guassianKernel=buildGaussianKernel(context, commandQueue, gscale, gussianKernelSize);
//    ScharrSeparableKernel scharrKernel=buildScharrSeparableKernel(context, 1.0, scharrKernelSize, false);

    std::vector<cl_int> histogram(nbins);
    ::cl::Buffer histogramBuffer(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, sizeof(cl_int)*histogram.size(), histogram.data());
    ::cl::Buffer histogramScratchBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int)*height*nbins);
    
    ::cl::Buffer maxBuffer(context, CL_MEM_READ_WRITE, sizeof(float));

    std::vector<::cl::Event> contrastWaitEvent(1);
    ::cl::Event hitogramMaxEvent;
    ::cl::Event hitogramEvent;
//    return computeKPercentile(context, commandQueue, src, width, height, perc, nbins, gaussian, magnitude, histogramBuffer, histogram, histogramScratchBuffer, 
//        guassianKernel, gussianKernelSize, scratch);

    float histogramMax;

    commandQueue.enqueueReadBuffer(maxBuffer, false, 0, sizeof(float), &histogramMax, &contrastWaitEvent, &hitogramMaxEvent);
    commandQueue.enqueueReadBuffer(histogramBuffer, false, 0, sizeof(cl_int)*histogram.size(), histogram.data(), &contrastWaitEvent, &hitogramEvent);

    hitogramMaxEvent.wait();
    hitogramEvent.wait();

    float contrast=computeContrast(histogram, histogramMax, perc);

    return contrast;
}

void computeKPercentile(::cl::Context &context, ::cl::CommandQueue &commandQueue, ::cl::Image2D &src, int width, int height, size_t nbins,
    ::cl::Image2D &gaussian, ::cl::Image2D &magnitude, ::cl::Buffer &histogramBuffer, ::cl::Buffer &histogramScratchBuffer,
    ::cl::Buffer &guassianKernel, int guassiankernelSize, ::cl::Image2D &scratch, ::cl::Image2D lx, ::cl::Image2D ly, ScharrSeparableKernel &scharrKernel, int scharrKernelSize,
    ::cl::Buffer &maxValue, std::vector<::cl::Event> *events, ::cl::Event &event)
{
    std::vector<::cl::Event> guassianWaitEvent(1);
    std::vector<::cl::Event> combinedWaitEvent(1);
    std::vector<::cl::Event> maxWaitEvent(1);
    
    // Perform the Gaussian convolution
//    gaussianSeparable(context, commandQueue, src, gaussian, width, height, gscale, nullptr, guassianEvent);
    gaussianSeparable(context, commandQueue, src, gaussian, width, height, guassianKernel, guassiankernelSize, scratch, events, guassianWaitEvent[0]);

//    {
//        guassianWaitEvent[0].wait();
//        std::string outputFile="../output/computeContrastGauss_cl.jpg";
//        saveImage2D(commandQueue, gaussian, outputFile);
//    }

#if 0
    std::vector<::cl::Event> derivativeEvent(2);

    // Compute the Gaussian derivatives Lx and Ly
    scharrSeparable(context, commandQueue, gaussian, width, height, 1, 1.0, false, false, lx, scharrKernel, scharrKernelSize, scratch, &guassianWaitEvent, derivativeEvent[0]);
    scharrSeparable(context, commandQueue, gaussian, width, height, 1, 1.0, true, false, ly, scharrKernel, scharrKernelSize, scratch, &guassianWaitEvent, derivativeEvent[1]);

    // Calculate magnitude
    calculateMagnitude(context, commandQueue, lx, ly, magnitude, width, height, &derivativeEvent, combinedWaitEvent[0]);
#else
    computeContrast(context, commandQueue, gaussian, width, height, magnitude, &guassianWaitEvent, combinedWaitEvent[0]);
#endif

    // Get the maximum from the magnitude
    calculateMax(context, commandQueue, magnitude, width, height, maxValue, &combinedWaitEvent, maxWaitEvent[0]);

    //Create scaled histogram using nbins and max of magnitude
//    float maxScale=1/hmax;
    calculateHistogram(context, commandQueue, magnitude, width, height, nbins, maxValue, histogramBuffer, histogramScratchBuffer,
        &maxWaitEvent, event);
}

float computeContrast(std::vector<int> &histogram, float hmax, float perc)
{
    size_t nbin=0, nelements=0, nthreshold=0, k=0;
    float kperc=0.0, modg=0.0, npoints=0.0;

    for(size_t i=0; i<histogram.size(); ++i)
        npoints+=histogram[i];

    size_t nbins=histogram.size();
    // Now find the perc of the histogram percentile
    nthreshold=(size_t)(npoints * perc);

    for(k=0; nelements < nthreshold && k < nbins; k++)
        nelements=nelements+histogram[k];

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
}

}}//namespace libAKAZE::cl