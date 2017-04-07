#include "benchmark_opencl.h"

#ifdef AKAZE_USE_OPENCL

#include "lib/convolve_cl.h"
#include "lib/filters_cl.h"

#include "timer/timer.hpp"

void benchmarkOpenCL_Gauss(::cl::Context &context, ::cl::CommandQueue &queue, Options &options, ::cl::Image2D &image, size_t width, size_t height, ::cl::Image2D &output, ::cl::Image2D &scratch, float sigma)
{
    int kernelSize;
    ::cl::Buffer gaussKernel=libAKAZE::cl::buildGaussianSeparableKernel(context, sigma, kernelSize);
    ::cl::Event event;
    std::vector<::cl::Event> events(1);
    double time;

    if(options.saveImages)
    {//clear out output image
        libAKAZE::cl::zeroImage(context, queue, output, nullptr, event);
        event.wait();
    }

    timer::Timer timer;

    libAKAZE::cl::separableConvolve(context, queue, image, width, height, gaussKernel, kernelSize, gaussKernel, kernelSize,
        1.0f, output, scratch, nullptr, events[0]);
    for(size_t i=1; i<options.iterations; ++i)
    {
        libAKAZE::cl::separableConvolve(context, queue, image, width, height, gaussKernel, kernelSize, gaussKernel, kernelSize,
            1.0f, output, scratch, &events, event);
        events[0]=event;
    }
    events[0].wait();

    time=timer.elapsedMs();
    std::cout<<"      separableConvolve "<<kernelSize<<"x"<<kernelSize<<" w/ scratch: "<<(time/options.iterations)<<"ms (Total: "<<time<<")"<<std::endl;

    if(options.saveImages)
    {
        std::ostringstream filename;

        filename<<"gaussFilter_"<<kernelSize<<"x"<<kernelSize<<".jpg";
        libAKAZE::cl::saveImage2D(queue, output, filename.str());
    }
}

void benchmarkOpenCL_Gauss_LocalXY(::cl::Context &context, ::cl::CommandQueue &queue, Options &options, ::cl::Image2D &image, size_t width, size_t height, ::cl::Image2D &output, float sigma)
{
    int kernelSize;
    ::cl::Buffer gaussKernel=libAKAZE::cl::buildGaussianSeparableKernel(context, sigma, kernelSize);
    ::cl::Event event;
    std::vector<::cl::Event> events(1);
    double time;

    if(options.saveImages)
    {//clear out output image
        libAKAZE::cl::zeroImage(context, queue, output, nullptr, event);
        event.wait();
    }

    timer::Timer timer;

    libAKAZE::cl::separableConvolve_localXY(context, queue, image, width, height, gaussKernel, gaussKernel, kernelSize,
        1.0f, output, nullptr, events[0]);
    for(size_t i=1; i<options.iterations; ++i)
    {
        libAKAZE::cl::separableConvolve_localXY(context, queue, image, width, height, gaussKernel, gaussKernel, kernelSize,
            1.0f, output, &events, event);
        events[0]=event;
    }
    events[0].wait();

    time=timer.elapsedMs();
    std::cout<<"      separableConvolve "<<kernelSize<<"x"<<kernelSize<<" w/ scratch: "<<(time/options.iterations)<<"ms (Total: "<<time<<")"<<std::endl;

    if(options.saveImages)
    {
        std::ostringstream filename;

        filename<<"gaussFilter_localXY"<<kernelSize<<"x"<<kernelSize<<".jpg";
        libAKAZE::cl::saveImage2D(queue, output, filename.str());
    }
}

void benchmarkOpenCL_Scharr(::cl::Context &context, ::cl::CommandQueue &queue, Options &options, ::cl::Image2D &image, size_t width, size_t height, ::cl::Image2D &output, ::cl::Image2D &scratch, int size)
{
    int kernelSize;
    libAKAZE::cl::ScharrSeparableKernel scharrKernel=libAKAZE::cl::buildScharrSeparableKernel(context, size, kernelSize, false);
    ::cl::Event event;
    std::vector<::cl::Event> events(1);
    double time;

    if(options.saveImages)
    {//clear out output image
        libAKAZE::cl::zeroImage(context, queue, output, nullptr, event);
        event.wait();
    }

    timer::Timer timer;

    libAKAZE::cl::separableConvolve(context, queue, image, width, height, scharrKernel.edge, kernelSize, scharrKernel.smooth, kernelSize,
        1.0f, output, scratch, nullptr, events[0]);
    for(size_t i=1; i<options.iterations; ++i)
    {
        libAKAZE::cl::separableConvolve(context, queue, image, width, height, scharrKernel.edge, kernelSize, scharrKernel.smooth, kernelSize,
            1.0f, output, scratch, &events, event);
        events[0]=event;
    }
    events[0].wait();

    time=timer.elapsedMs();
    std::cout<<"      separableConvolve "<<kernelSize<<"x"<<kernelSize<<" w/ scratch: "<<(time/options.iterations)<<"ms (Total: "<<time<<")"<<std::endl;

    if(options.saveImages)
    {
        std::ostringstream filename;

        filename<<"scharrFilter_"<<kernelSize<<"x"<<kernelSize<<".jpg";
        libAKAZE::cl::saveImage2D(queue, output, filename.str());
    }
}

void benchmarkOpenCL_Scharr_Local(::cl::Context &context, ::cl::CommandQueue &queue, Options &options, ::cl::Image2D &image, size_t width, size_t height, ::cl::Image2D &output, ::cl::Image2D &scratch, int size)
{
    int kernelSize;
    libAKAZE::cl::ScharrSeparableKernel scharrKernel=libAKAZE::cl::buildScharrSeparableKernel(context, size, kernelSize, false);
    ::cl::Event event;
    std::vector<::cl::Event> events(1);
    double time;

    if(options.saveImages)
    {//clear out output image
        libAKAZE::cl::zeroImage(context, queue, output, nullptr, event);
        event.wait();
    }

    timer::Timer timer;

    libAKAZE::cl::separableConvolve_local(context, queue, image, width, height, scharrKernel.edge, kernelSize, scharrKernel.smooth, kernelSize,
        1.0f, output, scratch, nullptr, events[0]);
    for(size_t i=1; i<options.iterations; ++i)
    {
        libAKAZE::cl::separableConvolve_local(context, queue, image, width, height, scharrKernel.edge, kernelSize, scharrKernel.smooth, kernelSize,
            1.0f, output, scratch, &events, event);
        events[0]=event;
    }
    events[0].wait();

    time=timer.elapsedMs();
    std::cout<<"      separableConvolve_local "<<kernelSize<<"x"<<kernelSize<<" w/ scratch: "<<(time/options.iterations)<<"ms (Total: "<<time<<")"<<std::endl;

    if(options.saveImages)
    {
        std::ostringstream filename;

        filename<<"scharrFilter_local_"<<kernelSize<<"x"<<kernelSize<<".jpg";
        libAKAZE::cl::saveImage2D(queue, output, filename.str());
    }
}

void benchmarkOpenCL_Scharr_LocalXY(::cl::Context &context, ::cl::CommandQueue &queue, Options &options, ::cl::Image2D &image, size_t width, size_t height, ::cl::Image2D &output, int size)
{
    int kernelSize;
    libAKAZE::cl::ScharrSeparableKernel scharrKernel=libAKAZE::cl::buildScharrSeparableKernel(context, size, kernelSize, false);
    ::cl::Event event;
    std::vector<::cl::Event> events(1);
    double time;
    
    if(options.saveImages)
    {//clear out output image
        libAKAZE::cl::zeroImage(context, queue, output, nullptr, event);
        event.wait();
    }

    timer::Timer timer;

    libAKAZE::cl::separableConvolve_localXY(context, queue, image, width, height, scharrKernel.edge, scharrKernel.smooth, kernelSize,
        1.0f, output, nullptr, events[0]);
    for(size_t i=1; i<options.iterations; ++i)
    {
        libAKAZE::cl::separableConvolve_localXY(context, queue, image, width, height, scharrKernel.edge, scharrKernel.smooth, kernelSize,
            1.0f, output, &events, event);
        events[0]=event;
    }
    events[0].wait();

    time=timer.elapsedMs();
    std::cout<<"      separableConvolve_localXY "<<kernelSize<<"x"<<kernelSize<<": "<<(time/options.iterations)<<"ms (Total: "<<time<<")"<<std::endl;

    if(options.saveImages)
    {
        std::ostringstream filename;

        filename<<"scharrFilter_localXY_"<<kernelSize<<"x"<<kernelSize<<".jpg";
        libAKAZE::cl::saveImage2D(queue, output, filename.str());
    }
}

void benchmarkOpenCLConvole(::cl::Context &context, ::cl::CommandQueue &queue, Options &options, ::cl::Image2D &image, size_t width, size_t height, ::cl::Image2D &output)
{
    std::cout<<"  Convolve:"<<std::endl;

    ::cl::Image2D scratch(context, CL_MEM_READ_WRITE, ::cl::ImageFormat(CL_R, CL_FLOAT), width, height);

    ::cl::Event event;

    libAKAZE::cl::zeroImage(context, queue, scratch, nullptr, event);
    event.wait();

    std::cout<<"    Gauss Kernel"<<std::endl;

    benchmarkOpenCL_Gauss(context, queue, options, image, width, height, output, scratch, 0.79f);
    benchmarkOpenCL_Gauss(context, queue, options, image, width, height, output, scratch, 1.09f);
    benchmarkOpenCL_Gauss(context, queue, options, image, width, height, output, scratch, 1.39f);
    benchmarkOpenCL_Gauss(context, queue, options, image, width, height, output, scratch, 1.69f);
    benchmarkOpenCL_Gauss(context, queue, options, image, width, height, output, scratch, 1.99f);
    
    benchmarkOpenCL_Gauss_LocalXY(context, queue, options, image, width, height, output, 0.79f);
    benchmarkOpenCL_Gauss_LocalXY(context, queue, options, image, width, height, output, 1.09f);
    benchmarkOpenCL_Gauss_LocalXY(context, queue, options, image, width, height, output, 1.39f);
    benchmarkOpenCL_Gauss_LocalXY(context, queue, options, image, width, height, output, 1.69f);
    benchmarkOpenCL_Gauss_LocalXY(context, queue, options, image, width, height, output, 1.99f);
    
    std::cout<<"    Scharr Kernel"<<std::endl;

    benchmarkOpenCL_Scharr(context, queue, options, image, width, height, output, scratch, 1);
    benchmarkOpenCL_Scharr(context, queue, options, image, width, height, output, scratch, 2);
    benchmarkOpenCL_Scharr(context, queue, options, image, width, height, output, scratch, 3);
    benchmarkOpenCL_Scharr(context, queue, options, image, width, height, output, scratch, 4);

    benchmarkOpenCL_Scharr_Local(context, queue, options, image, width, height, output, scratch, 1);
    benchmarkOpenCL_Scharr_Local(context, queue, options, image, width, height, output, scratch, 2);
    benchmarkOpenCL_Scharr_Local(context, queue, options, image, width, height, output, scratch, 3);
    benchmarkOpenCL_Scharr_Local(context, queue, options, image, width, height, output, scratch, 4);

    benchmarkOpenCL_Scharr_LocalXY(context, queue, options, image, width, height, output, 1);
    benchmarkOpenCL_Scharr_LocalXY(context, queue, options, image, width, height, output, 2);
    benchmarkOpenCL_Scharr_LocalXY(context, queue, options, image, width, height, output, 3);
    benchmarkOpenCL_Scharr_LocalXY(context, queue, options, image, width, height, output, 4);
}

void benchmarkOpenCLAKAZE(::cl::Context &context, ::cl::CommandQueue &queue, Options &options, RowMatrixXf &image)
{
    timer::Timer timer;
    double totalTime;
    libAKAZE::Options akaze_options;

    akaze_options.img_width=image.cols();
    akaze_options.img_height=image.rows();

    libAKAZE::cl::AKAZE evolution(context, queue, akaze_options);

    evolution.initOpenCL(); //gets some items loaded early

    timer.reset();

    evolution.Create_Nonlinear_Scale_Space(image);
    evolution.Feature_Detection();
    evolution.Compute_Descriptors();

    totalTime=timer.elapsedMs();

    std::vector<libAKAZE::Keypoint> kpts;
    libAKAZE::Descriptors desc;

    evolution.getKeypoints(kpts);
    evolution.getDescriptors(desc);

    std::cout<<"Number of points: "<<kpts.size()<<std::endl;
    evolution.Show_Computation_Times();
    std::cout<<"Total Time: "<<totalTime<<std::endl;
}

void benchmarkOpenCLDevice(::cl::Context &openClContext, RowMatrixXf &image, Options &options)
{
    ::cl::CommandQueue commandQueue(openClContext);

    ::cl::Image2D imageCL(openClContext, CL_MEM_READ_WRITE, ::cl::ImageFormat(CL_R, CL_FLOAT), image.cols(), image.rows());
    ::cl::Image2D outputCL(openClContext, CL_MEM_READ_WRITE, ::cl::ImageFormat(CL_R, CL_FLOAT), image.cols(), image.rows());

    std::vector<::cl::Event> bufferEvents(2);
    size_t imageSize=image.cols()*image.rows()*sizeof(float);
    ::cl::size_t<3> origin;
    ::cl::size_t<3> region;

    origin[0]=0;
    origin[1]=0;
    origin[2]=0;

    region[0]=image.cols();
    region[1]=image.rows();
    region[2]=1;

    commandQueue.enqueueWriteImage(imageCL, CL_FALSE, origin, region, 0, 0, image.data(), nullptr, &bufferEvents[0]);
    libAKAZE::cl::zeroImage(openClContext, commandQueue, outputCL, nullptr, bufferEvents[1]); //force image to be created;

    ::cl::WaitForEvents(bufferEvents);

//    benchmarkOpenCLConvole(openClContext, commandQueue, options, imageCL, image.cols(), image.rows(), outputCL);
    benchmarkOpenCLAKAZE(openClContext, commandQueue, options, image);
}

#endif//AKAZE_USE_OPENCL