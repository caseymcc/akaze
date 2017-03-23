/**
 * @file AKAZE_cl.cpp
 * @brief Main class for opencl implementation of AKAZE
 */

#include "AKAZE_cl.h"
#include <cstdio>  //%%%%
#include <iostream>

#include "filters_cl.h"
//#include "cimg/CImg.h"
#include "timer/timer.hpp"
#include "utils.h"

#include "openClContext.h"

using namespace std;

namespace libAKAZE
{
namespace cl
{

/* ************************************************************************* */
//AKAZE::AKAZE(cl_context openclContext, cl_command_queue commandQueue, const Options& options):
AKAZE::AKAZE(::cl::Context openclContext, ::cl::CommandQueue commandQueue, const Options& options):
    options_(options),
    openclContext_(openclContext),
    commandQueue_(commandQueue)
{
    saveImages_=false;
    saveCsv_=false;
    reordering_=true;
    width_=0;
    height_=0;
//    openclContext_=openclContext;

    Allocate_Memory_Evolution();
}

/* ************************************************************************* */
AKAZE::~AKAZE()
{
}

void AKAZE::initOpenCL()
{
    ::cl::Kernel kernel;

    //init all kernels so they a ready for the calls
    kernel=getKernel(openclContext_, "separableConvolveXImage2D", "lib/kernels/convolve.cl");
    kernel=getKernel(openclContext_, "separableConvolveYImage2D", "lib/kernels/convolve.cl");
    kernel=getKernel(openclContext_, "separableConvolveXImage2DBuffer", "lib/kernels/convolve.cl");
    kernel=getKernel(openclContext_, "separableConvolveXBuffer", "lib/kernels/convolve.cl");
    kernel=getKernel(openclContext_, "separableConvolveYBuffer", "lib/kernels/convolve.cl");
    kernel=getKernel(openclContext_, "magnitude", "lib/kernels/convolve.cl");
    kernel=getKernel(openclContext_, "rowMax", "lib/kernels/convolve.cl");
    kernel=getKernel(openclContext_, "histogramRows", "lib/kernels/convolve.cl");
    kernel=getKernel(openclContext_, "histogramCombine", "lib/kernels/convolve.cl");
    kernel=getKernel(openclContext_, "pmG1", "lib/kernels/convolve.cl");
    kernel=getKernel(openclContext_, "pmG2", "lib/kernels/convolve.cl");
    kernel=getKernel(openclContext_, "weickert", "lib/kernels/convolve.cl");
    kernel=getKernel(openclContext_, "charbonnier", "lib/kernels/convolve.cl");
    kernel=getKernel(openclContext_, "nldStepScalar", "lib/kernels/convolve.cl");
    kernel=getKernel(openclContext_, "determinantHessianBuffer", "lib/kernels/convolve.cl");
    kernel=getKernel(openclContext_, "findExtremaBuffer", "lib/kernels/convolve.cl");
    kernel=getKernel(openclContext_, "consolidateKeypoints", "lib/kernels/convolve.cl");
    kernel=getKernel(openclContext_, "subPixelRefinement", "lib/kernels/convolve.cl");
    kernel=getKernel(openclContext_, "computeOrientation", "lib/kernels/convolve.cl");
    kernel=getKernel(openclContext_, "getMLDBDescriptor", "lib/kernels/convolve.cl");
}

void AKAZE::Allocate_Memory_Evolution()
{
    generateEvolution(options_, evolution_);

    size_t bufferSize=0;
    
    for(int i=0; i<evolution_.size(); ++i)
    {
        EvolutionCL &step=evolution_[i];

        bufferSize+=step.width*step.height;
    }

    evolutionImage_=::cl::Buffer(openclContext_, CL_MEM_READ_WRITE, bufferSize*sizeof(cl_float));
    evolutionDx_=::cl::Buffer(openclContext_, CL_MEM_READ_WRITE, bufferSize*sizeof(cl_float));
    evolutionDy_=::cl::Buffer(openclContext_, CL_MEM_READ_WRITE, bufferSize*sizeof(cl_float));
    evolutionDxx_=::cl::Buffer(openclContext_, CL_MEM_READ_WRITE, bufferSize*sizeof(cl_float));
    evolutionDxy_=::cl::Buffer(openclContext_, CL_MEM_READ_WRITE, bufferSize*sizeof(cl_float));
    evolutionDyy_=::cl::Buffer(openclContext_, CL_MEM_READ_WRITE, bufferSize*sizeof(cl_float));
    evolutionDet_=::cl::Buffer(openclContext_, CL_MEM_READ_WRITE, bufferSize*sizeof(cl_float));
    evolutionInfo_=::cl::Buffer(openclContext_, CL_MEM_READ_WRITE, evolution_.size()*sizeof(EvolutionInfo));

    extremaMap_=::cl::Buffer(openclContext_, CL_MEM_READ_WRITE, bufferSize*sizeof(ExtremaMap));

    std::vector<EvolutionInfo> evolutionInfo(evolution_.size());

    size_t offset=0;
    for(int i=0; i<evolution_.size(); ++i)
    {
        EvolutionCL &step=evolution_[i];
        EvolutionInfo &info=evolutionInfo[i];

        step.offset=offset;
        step.image=::cl::Image2D(openclContext_, CL_MEM_READ_WRITE, ::cl::ImageFormat(CL_R, CL_FLOAT), step.width, step.height);
        step.smooth=::cl::Image2D(openclContext_, CL_MEM_READ_WRITE, ::cl::ImageFormat(CL_R, CL_FLOAT), step.width, step.height);
        step.lx=::cl::Image2D(openclContext_, CL_MEM_READ_WRITE, ::cl::ImageFormat(CL_R, CL_FLOAT), step.width, step.height);
        step.ly=::cl::Image2D(openclContext_, CL_MEM_READ_WRITE, ::cl::ImageFormat(CL_R, CL_FLOAT), step.width, step.height);
        step.flow=::cl::Image2D(openclContext_, CL_MEM_READ_WRITE, ::cl::ImageFormat(CL_R, CL_FLOAT), step.width, step.height);
        step.step=::cl::Image2D(openclContext_, CL_MEM_READ_WRITE, ::cl::ImageFormat(CL_R, CL_FLOAT), step.width, step.height);

        step.dx=evolutionDx_;
        step.dy=evolutionDy_;
//        step.lxx=::cl::Image2D(openclContext_, CL_MEM_READ_WRITE, ::cl::ImageFormat(CL_R, CL_FLOAT), step.width, step.height);
//        step.lxy=::cl::Image2D(openclContext_, CL_MEM_READ_WRITE, ::cl::ImageFormat(CL_R, CL_FLOAT), step.width, step.height);
//        step.lyy=::cl::Image2D(openclContext_, CL_MEM_READ_WRITE, ::cl::ImageFormat(CL_R, CL_FLOAT), step.width, step.height);
        step.dxx=evolutionDxx_;
        step.dxy=evolutionDxy_;
        step.dyy=evolutionDyy_;

//        step.det=::cl::Image2D(openclContext_, CL_MEM_READ_WRITE, ::cl::ImageFormat(CL_R, CL_FLOAT), step.width, step.height);
        step.det=evolutionDet_;

        step.scratch=::cl::Image2D(openclContext_, CL_MEM_READ_WRITE, ::cl::ImageFormat(CL_R, CL_FLOAT), step.width, step.height);

        info.width=step.width;
        info.height=step.height;
        info.offset=offset;
        info.sigma=step.sigma;
        info.pointSize=step.sigma*options_.derivative_factor;
        info.octave=step.octave;

        offset+=step.width*step.height;
    }

    ::cl::Event event;

    commandQueue_.enqueueWriteBuffer(evolutionInfo_, CL_FALSE, 0, evolutionInfo.size()*sizeof(cl::EvolutionInfo), evolutionInfo.data(), nullptr, &event);

    // Allocate memory for the number of cycles and time steps
    for(size_t i=1; i < evolution_.size(); i++)
    {
        int naux=0;
        vector<float> tau;
        float ttime=0.0;

        ttime=evolution_[i].time-evolution_[i-1].time;
        naux=fed_tau_by_process_time(ttime, 1, 0.25, reordering_, tau);
        nsteps_.push_back(naux);
        tsteps_.push_back(tau);
        ncycles_++;
    }

    event.wait();

    contrastGuassian_=buildGaussianKernel(context, commandQueue, 1.0, contrastGuassianSize_);
    contrastScharr_=buildScharrSeparableKernel(context, 1, contrastScharrSize_, normalize);

    histogram_.resize(nbins);
    histogramBuffer_=::cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*histogram.size(), histogram.data());
    histogramScratchBuffer_=::cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_int)*evolution_[0].height*options_.kcontrast_nbins);
}

int AKAZE::Create_Nonlinear_Scale_Space(const RowMatrixXf &image)
{
    if(evolution_.size()==0)
    {
        cerr<<"Error generating the nonlinear scale space!!"<<endl;
        cerr<<"Firstly you need to call AKAZE::Allocate_Memory_Evolution()"<<endl;
        return -1;
    }

//    if(image.col != width_)

    //copy image to the first level of the evolution
    ::cl::size_t<3> origin;
    ::cl::size_t<3> region;
    ::cl::Event imageEvent;
    ::cl::Event guassEvent;
    ::cl::Event copyEvent;
    std::vector<::cl::Event> events;
    timer::Timer timer;

    origin[0]=0;
    origin[1]=0;
    origin[2]=0;

    region[0]=image.cols();
    region[1]=image.rows();
    region[2]=1;

    commandQueue_.enqueueWriteImage(evolution_[0].image, CL_FALSE, origin, region, 0, 0, (void *)image.data(), nullptr, &imageEvent);
    events.push_back(imageEvent);

//    imageEvent.wait();

    gaussianSeparable(openclContext_, commandQueue_, evolution_[0].image, evolution_[0].smooth, image.cols(), image.rows(), options_.soffset, &events, guassEvent);
//    guassEvent.wait();
    events.push_back(guassEvent);
    commandQueue_.enqueueCopyImage(evolution_[0].smooth, evolution_[0].image, origin, origin, region, &events, &copyEvent);

//    float kcontrast=computeKPercentile(openclContext_, commandQueue_, evolution_[0].image, options_.kcontrast_percentile, 1.0, options_.kcontrast_nbins, 0, 0);
    float kcontrast=computeKPercentile(openclContext_, commandQueue_, evolution_[0].image, evolution_[0].width, evolution_[0].height, options_.kcontrast_percentile, options_.kcontrast_nbins,
        contrastGuassianScratch_, contrastMagnitudeScratch_, histogramBuffer_, histogram_, histogramScratchBuffer_, contrastGuassian_, contrastGuassianSize_, contrastScharr_, contrastScharrSize_,
        evolution_[0].scratch);

    timing_.kcontrast=timer.elapsedMs();

    copyEvent.wait();

    for(size_t i=1; i<evolution_.size(); i++)
    {
        std::vector<::cl::Event> copyImageWaitEvent;

        if(evolution_[i].octave > evolution_[i-1].octave)
        {
            ::cl::Event copyImageEvent;

            linearSample(openclContext_, commandQueue_, evolution_[i-1].image, evolution_[i].image, evolution_[i].width, evolution_[i].height, nullptr, copyImageEvent);
            copyImageWaitEvent.push_back(copyImageEvent);

            kcontrast=kcontrast*0.75;
        }
        else
        {
            ::cl::Event copyImageEvent;

//            evolution_[i].image=evolution_[i-1].image;
            region[0]=evolution_[i].width;
            region[1]=evolution_[i].height;
            region[2]=1;

            commandQueue_.enqueueCopyImage(evolution_[i-1].image, evolution_[i].image, origin, origin, region, nullptr, &copyImageEvent);
            copyImageWaitEvent.push_back(copyImageEvent);
        }

        ::cl::Event gaussEvent;

//        if(!copyImageWaitEvent.empty())
            gaussianSeparable(openclContext_, commandQueue_, evolution_[i].image, evolution_[i].smooth, evolution_[i].width, evolution_[i].height, 1.0, &copyImageWaitEvent, gaussEvent);
//        else
//            gaussianSeparable(openclContext_, commandQueue_, evolution_[i].image, evolution_[i].smooth, evolution_[i].width, evolution_[i].height, 1.0, nullptr, gaussEvent);

        std::vector<::cl::Event>gaussWaitEvent={gaussEvent};
        ::cl::Event derivativeXEvent;
        ::cl::Event derivativeYEvent;

        // Compute the Gaussian derivatives lx and ly
        scharrSeparable(openclContext_, commandQueue_, evolution_[i].smooth, evolution_[i].width, evolution_[i].height, 1, 1.0, false, false, evolution_[i].lx, &gaussWaitEvent, derivativeXEvent);
        scharrSeparable(openclContext_, commandQueue_, evolution_[i].smooth, evolution_[i].width, evolution_[i].height, 1, 1.0, true, false, evolution_[i].ly, &gaussWaitEvent, derivativeYEvent);

        std::vector<::cl::Event>derivativeWaitEvent={derivativeXEvent, derivativeYEvent};
        ::cl::Event diffusivityEvent;

        switch(options_.diffusivity)
        {
        case PM_G1:
            pmG1(openclContext_, commandQueue_, evolution_[i].lx, evolution_[i].ly, evolution_[i].flow, evolution_[i].width, evolution_[i].height, kcontrast, &derivativeWaitEvent, diffusivityEvent);
            break;
        case PM_G2:
            pmG2(openclContext_, commandQueue_, evolution_[i].lx, evolution_[i].ly, evolution_[i].flow, evolution_[i].width, evolution_[i].height, kcontrast, &derivativeWaitEvent, diffusivityEvent);
            break;
        case WEICKERT:
            break;
        case CHARBONNIER:
            break;
        }

        std::vector<::cl::Event>stepWait={diffusivityEvent};

        for(int j=0; j<nsteps_[i-1]; j++)
        {
            ::cl::Event stepEvent;

            nldStepScalar(openclContext_, commandQueue_, evolution_[i].image, evolution_[i].flow, evolution_[i].step, evolution_[i].width, evolution_[i].height, tsteps_[i-1][j], &stepWait, stepEvent);

            //swap image references for next loop
            ::cl::Image2D tempImage=evolution_[i].image;
            evolution_[i].image=evolution_[i].step;
            evolution_[i].step=tempImage;

//            stepEvent.wait();
            stepWait[0]=stepEvent;
        }

//        stepWait[0].wait();
    }


    //copy final images over to buffer likely better to alter evolution building to use the buffer instead, but as a shortcut this will work for now)
    std::vector<::cl::Event> imageEvents(evolution_.size());

    for(size_t i=0; i<evolution_.size(); i++)
    {
        region[0]=evolution_[i].width;
        region[1]=evolution_[i].height;

        commandQueue_.enqueueCopyImageToBuffer(evolution_[i].image, evolutionImage_, origin, region, evolution_[i].offset*sizeof(cl_float), nullptr, &imageEvents[i]);
    }
    ::cl::WaitForEvents(imageEvents);

    timing_.evolution=timer.elapsedMs();
    timing_.scale=timing_.evolution+timing_.kcontrast;

    if(saveImages_)
    {
        std::string outputFile;

        for(size_t i=0; i<evolution_.size(); ++i)
        {
            outputFile="../output/evolution_"+to_formatted_string(i, 2)+"_cl.jpg";
            saveImage2D(commandQueue_, evolution_[i].image, outputFile);
            
            outputFile="../output/evolutionSmooth_"+to_formatted_string(i, 2)+"_cl.jpg";
            saveImage2D(commandQueue_, evolution_[i].smooth, outputFile);

            outputFile="../output/evolutionFlow_"+to_formatted_string(i, 2)+"_cl.jpg";
            saveImage2D(commandQueue_, evolution_[i].flow, outputFile);
        }
    }

    if(saveCsv_)
    {
        std::string outputFile;

        for(size_t i=0; i<evolution_.size(); ++i)
        {
            outputFile="../output/evolution_"+to_formatted_string(i, 2)+"_cl.csv";
            saveImage2DCsv(commandQueue_, evolution_[i].image, outputFile);

            outputFile="../output/evolutionSmooth_"+to_formatted_string(i, 2)+"_cl.csv";
            saveImage2DCsv(commandQueue_, evolution_[i].smooth, outputFile);

            outputFile="../output/evolutionFlow_"+to_formatted_string(i, 2)+"_cl.csv";
            saveImage2DCsv(commandQueue_, evolution_[i].flow, outputFile);
        }
    }

    return -1;
}

void AKAZE::Load_Nonlinear_Scale_Space(std::string &directory)
{
    std::string outputFile;

    std::vector<::cl::Event> imageEvents(evolution_.size());
    std::vector<::cl::Event> smoothEvents(evolution_.size());

    for(size_t i=0; i<evolution_.size(); ++i)
    {
        outputFile=directory+"/evolution_"+to_formatted_string(i, 2)+"_cl.dat";
        loadImage2DData(commandQueue_, evolution_[i].image, outputFile, imageEvents[i]);

        outputFile=directory+"/evolutionSmooth_"+to_formatted_string(i, 2)+"_cl.dat";
        loadImage2DData(commandQueue_, evolution_[i].smooth, outputFile, smoothEvents[i]);
    }

    ::cl::WaitForEvents(imageEvents);
    ::cl::WaitForEvents(smoothEvents);

    ::cl::size_t<3> origin;
	::cl::size_t<3> region;

    origin[0]=0;
    origin[1]=0;
    origin[2]=0;
	region[2]=1;

    //copy final images over to buffer likely better to alter evolution building to use the buffer instead, but as a shortcut this will work for now)
    std::vector<::cl::Event> imageBufferEvents(evolution_.size());

    for(size_t i=0; i<evolution_.size(); i++)
    {
        region[0]=evolution_[i].width;
        region[1]=evolution_[i].height;

        commandQueue_.enqueueCopyImageToBuffer(evolution_[i].image, evolutionImage_, origin, region, evolution_[i].offset*sizeof(cl_float), nullptr, &imageBufferEvents[i]);
    }
    ::cl::WaitForEvents(imageBufferEvents);
}

void AKAZE::Save_Nonlinear_Scale_Space(std::string &directory)
{
    std::string outputFile;

    for(size_t i=0; i<evolution_.size(); ++i)
    {
        outputFile=directory+"/evolution_"+to_formatted_string(i, 2)+"_cl.dat";
        saveImage2DData(commandQueue_, evolution_[i].image, outputFile);

        outputFile=directory+"/evolutionSmooth_"+to_formatted_string(i, 2)+"_cl.dat";
        saveImage2DData(commandQueue_, evolution_[i].smooth, outputFile);
    }
}

void AKAZE::Feature_Detection(std::vector<libAKAZE::Keypoint> &kpts)
{
    kpts.clear();

    timer::Timer timer;

    Compute_Determinant_Hessian_Response();
    Find_Scale_Space_Extrema(kpts);
//    Do_Subpixel_Refinement(kpts);
    timing_.detector=timer.elapsedMs();
}

void AKAZE::Compute_Multiscale_Derivatives(std::vector<std::vector<::cl::Event>> &evolutionEvents)
{
    for(int i=0; i < (int)(evolution_.size()); i++)
    {
        float ratio=pow(2.f, (float)evolution_[i].octave);
        int sigma_size_=fRound(evolution_[i].sigma * options_.derivative_factor/ratio);

        ::cl::Event lxEvent;
        ::cl::Event lyEvent;
        
        scharrSeparable(openclContext_, commandQueue_, evolution_[i].smooth, evolution_[i].width, evolution_[i].height, sigma_size_, sigma_size_, false, true, evolution_[i].dx, evolution_[i].offset, nullptr, lxEvent);
        scharrSeparable(openclContext_, commandQueue_, evolution_[i].smooth, evolution_[i].width, evolution_[i].height, sigma_size_, sigma_size_, true, true, evolution_[i].dy, evolution_[i].offset, nullptr, lyEvent);

        std::vector<::cl::Event> lxCompleteEvent={lxEvent};
        std::vector<::cl::Event> lyCompleteEvent={lyEvent};
        std::vector<::cl::Event> events(3);

        scharrSeparable(openclContext_, commandQueue_, evolution_[i].dx, evolution_[i].offset, evolution_[i].width, evolution_[i].height, sigma_size_, sigma_size_, false, true, evolution_[i].dxx, evolution_[i].offset, &lxCompleteEvent, events[0]);
        scharrSeparable(openclContext_, commandQueue_, evolution_[i].dx, evolution_[i].offset, evolution_[i].width, evolution_[i].height, sigma_size_, sigma_size_, true, true, evolution_[i].dxy, evolution_[i].offset, &lxCompleteEvent, events[1]);
        scharrSeparable(openclContext_, commandQueue_, evolution_[i].dy, evolution_[i].offset, evolution_[i].width, evolution_[i].height, sigma_size_, sigma_size_, true, true, evolution_[i].dyy, evolution_[i].offset, &lyCompleteEvent, events[2]);

        evolutionEvents.push_back(events);
    }

    if(saveImages_)
    {
        std::string outputFile;

        for(int i=0; i<(int)(evolution_.size()); i++)
        {
            ::cl::WaitForEvents(evolutionEvents[i]);

            outputFile="../output/evolutionLx_"+to_formatted_string(i, 2)+"_cl.jpg";
            saveBufferAsImage(commandQueue_, evolution_[i].dx, evolution_[i].offset, evolution_[i].width, evolution_[i].height, outputFile.c_str());

            outputFile="../output/evolutionLy_"+to_formatted_string(i, 2)+"_cl.jpg";
            saveBufferAsImage(commandQueue_, evolution_[i].dy, evolution_[i].offset, evolution_[i].width, evolution_[i].height, outputFile.c_str());

            outputFile="../output/evolutionLxx_"+to_formatted_string(i, 2)+"_cl.jpg";
            saveBufferAsImage(commandQueue_, evolution_[i].dxx, evolution_[i].offset, evolution_[i].width, evolution_[i].height, outputFile.c_str());

            outputFile="../output/evolutionLxy_"+to_formatted_string(i, 2)+"_cl.jpg";
            saveBufferAsImage(commandQueue_, evolution_[i].dxy, evolution_[i].offset, evolution_[i].width, evolution_[i].height, outputFile.c_str());

            outputFile="../output/evolutionLyy_"+to_formatted_string(i, 2)+"_cl.jpg";
            saveBufferAsImage(commandQueue_, evolution_[i].dyy, evolution_[i].offset, evolution_[i].width, evolution_[i].height, outputFile.c_str());
        }
    }

    if(saveCsv_)
    {
        std::string outputFile;

        for(int i=0; i<(int)(evolution_.size()); i++)
        {
            ::cl::WaitForEvents(evolutionEvents[i]);

            outputFile="../output/evolutionLx_"+to_formatted_string(i, 2)+"_cl.csv";
            saveBufferCsv(commandQueue_, evolution_[i].dx, evolution_[i].offset, evolution_[i].width, evolution_[i].height, outputFile.c_str());

            outputFile="../output/evolutionLy_"+to_formatted_string(i, 2)+"_cl.csv";
            saveBufferCsv(commandQueue_, evolution_[i].dy, evolution_[i].offset, evolution_[i].width, evolution_[i].height, outputFile.c_str());

            outputFile="../output/evolutionLxx_"+to_formatted_string(i, 2)+"_cl.csv";
            saveBufferCsv(commandQueue_, evolution_[i].dxx, evolution_[i].offset, evolution_[i].width, evolution_[i].height, outputFile.c_str());

            outputFile="../output/evolutionLxy_"+to_formatted_string(i, 2)+"_cl.csv";
            saveBufferCsv(commandQueue_, evolution_[i].dxy, evolution_[i].offset, evolution_[i].width, evolution_[i].height, outputFile.c_str());

            outputFile="../output/evolutionLyy_"+to_formatted_string(i, 2)+"_cl.csv";
            saveBufferCsv(commandQueue_, evolution_[i].dyy, evolution_[i].offset, evolution_[i].width, evolution_[i].height, outputFile.c_str());
        }
    }
//    timing_.derivatives=timer.elapsedMs();
}

void AKAZE::Load_Derivatives(std::string &directory)
{
    std::string outputFile;
    std::vector<::cl::Event> events(5*evolution_.size());
    size_t index=0;

    size_t bufferSize=0;
    for(size_t i=0; i<evolution_.size(); ++i)
    {
        bufferSize+=evolution_[i].width*evolution_[i].height;
    }

    std::vector<float> dxBuffer(bufferSize);
    std::vector<float> dyBuffer(bufferSize);
    std::vector<float> dxxBuffer(bufferSize);
    std::vector<float> dxyBuffer(bufferSize);
    std::vector<float> dyyBuffer(bufferSize);

    for(size_t i=0; i<evolution_.size(); ++i)
    {
        size_t size=evolution_[i].width*evolution_[i].height*sizeof(float);

        outputFile=directory+"/evolutionDx_"+to_formatted_string(i, 2)+"_cl.dat";
        loadBufferData(commandQueue_, evolution_[i].dx, evolution_[i].offset, &dxBuffer[evolution_[i].offset], size, outputFile, &events[index++]);
        outputFile=directory+"/evolutionDy_"+to_formatted_string(i, 2)+"_cl.dat";
        loadBufferData(commandQueue_, evolution_[i].dy, evolution_[i].offset, &dyBuffer[evolution_[i].offset], size, outputFile, &events[index++]);
        outputFile=directory+"/evolutionDxx_"+to_formatted_string(i, 2)+"_cl.dat";
        loadBufferData(commandQueue_, evolution_[i].dxx, evolution_[i].offset, &dxxBuffer[evolution_[i].offset], size, outputFile, &events[index++]);
        outputFile=directory+"/evolutionDxy_"+to_formatted_string(i, 2)+"_cl.dat";
        loadBufferData(commandQueue_, evolution_[i].dxy, evolution_[i].offset, &dxyBuffer[evolution_[i].offset], size, outputFile, &events[index++]);
        outputFile=directory+"/evolutionDyy_"+to_formatted_string(i, 2)+"_cl.dat";
        loadBufferData(commandQueue_, evolution_[i].dyy, evolution_[i].offset, &dyyBuffer[evolution_[i].offset], size, outputFile, &events[index++]);
    }

    ::cl::WaitForEvents(events);
}

void AKAZE::Save_Derivatives(std::string &directory)
{
    std::string outputFile;

    for(size_t i=0; i<evolution_.size(); ++i)
    {
        outputFile=directory+"/evolutionDx_"+to_formatted_string(i, 2)+"_cl.dat";
        saveBufferData(commandQueue_, evolution_[i].dx, evolution_[i].offset, evolution_[i].width, evolution_[i].height, outputFile);
        outputFile=directory+"/evolutionDy_"+to_formatted_string(i, 2)+"_cl.dat";
        saveBufferData(commandQueue_, evolution_[i].dy, evolution_[i].offset, evolution_[i].width, evolution_[i].height, outputFile);
        outputFile=directory+"/evolutionDxx_"+to_formatted_string(i, 2)+"_cl.dat";
        saveBufferData(commandQueue_, evolution_[i].dxx, evolution_[i].offset, evolution_[i].width, evolution_[i].height, outputFile);
        outputFile=directory+"/evolutionDxy_"+to_formatted_string(i, 2)+"_cl.dat";
        saveBufferData(commandQueue_, evolution_[i].dxy, evolution_[i].offset, evolution_[i].width, evolution_[i].height, outputFile);
        outputFile=directory+"/evolutionDyy_"+to_formatted_string(i, 2)+"_cl.dat";
        saveBufferData(commandQueue_, evolution_[i].dyy, evolution_[i].offset, evolution_[i].width, evolution_[i].height, outputFile);
    }
}

void AKAZE::Compute_Determinant_Hessian_Response()
{
    std::vector<std::vector<::cl::Event>> completeEvents;
    timer::Timer timer;

    Compute_Multiscale_Derivatives(completeEvents);

    std::vector<::cl::Event> events(evolution_.size());

    for(size_t i=0; i<evolution_.size(); ++i)
    {
        ::cl::Event event;

        determinantHessian(openclContext_, commandQueue_, evolution_[i].dxx, evolution_[i].offset, evolution_[i].dyy, evolution_[i].offset, evolution_[i].dxy, evolution_[i].offset, 
            evolution_[i].width, evolution_[i].height, evolution_[i].det, evolution_[i].offset, &completeEvents[i], events[i]);
    }

    ::cl::WaitForEvents(events);
    timing_.derivatives=timer.elapsedMs();

    if(saveImages_)
    {
        ::cl::WaitForEvents(events);
        std::string outputFile;

        for(size_t i=0; i<evolution_.size(); ++i)
        {
            outputFile="../output/evolutionDet_"+to_formatted_string(i, 2)+"_cl.jpg";
            saveBufferAsImage(commandQueue_, evolution_[i].det, evolution_[i].offset, evolution_[i].width, evolution_[i].height, outputFile, options_.dthreshold);
        }
    }
    if(saveCsv_)
    {
        ::cl::WaitForEvents(events);
        std::string outputFile;

        for(size_t i=0; i<evolution_.size(); ++i)
        {
            outputFile="../output/evolutionDet_"+to_formatted_string(i, 2)+"_cl.csv";
            saveBufferCsv(commandQueue_, evolution_[i].det,evolution_[i].offset, evolution_[i].width, evolution_[i].height, outputFile);
        }
    }
}

void AKAZE::Load_Determinant_Hessian_Response(std::string &directory)
{
    std::string outputFile;

    for(size_t i=0; i<evolution_.size(); ++i)
    {
        outputFile=directory+"/evolutionDet_"+to_formatted_string(i, 2)+"_cl.dat";
        loadBufferData(commandQueue_, evolution_[i].det, evolution_[i].offset, outputFile);
    }
}

void AKAZE::Save_Determinant_Hessian_Response(std::string &directory)
{
    std::string outputFile;

    for(size_t i=0; i<evolution_.size(); ++i)
    {
        outputFile=directory+"/evolutionDet_"+to_formatted_string(i, 2)+"_cl.dat";
        saveBufferData(commandQueue_, evolution_[i].det, evolution_[i].offset, evolution_[i].width, evolution_[i].height, outputFile);
    }
}

void AKAZE::saveDebug()
{
	std::string outputFile;

	for (size_t i = 0; i<evolution_.size(); ++i)
	{
		outputFile = "../output/evolutionDetLoad_" + to_formatted_string(i, 2) + "_cl.csv";
		saveBufferCsv(commandQueue_, evolution_[i].det, evolution_[i].offset, evolution_[i].width, evolution_[i].height, outputFile);
	}
}

void AKAZE::Find_Scale_Space_Extrema(std::vector<libAKAZE::Keypoint> &kpts)
{
    Find_Scale_Space_Extrema();
    getKeypoints(kpts);
}

void AKAZE::Find_Scale_Space_Extrema()
{
    size_t extremaMapWidth=evolution_[0].width;
    size_t extremaMapHeight=evolution_[0].height;
    float descriptorMax;
    size_t keypointStructSize=sizeof(cl::Keypoint);
    timer::Timer timer;

    if(options_.descriptor==SURF_UPRIGHT||options_.descriptor==SURF||
        options_.descriptor==MLDB_UPRIGHT||options_.descriptor==MLDB)
    {
        descriptorMax=10.0 * sqrtf(2.0);
    }
    else if(options_.descriptor==MSURF_UPRIGHT||
        options_.descriptor==MSURF)
    {
        descriptorMax=12.0 * sqrtf(2.0);
    }

    std::vector<::cl::Event> events(1);
    ::cl::Event event;
    float threashold=max(options_.dthreshold, options_.min_dthreshold);
    
    ::cl::Buffer keypointsCountBuffer(openclContext_, CL_MEM_READ_WRITE, 1*sizeof(int));
    std::vector<int> border(evolution_.size());

    float pointSize=evolution_[0].sigma*options_.derivative_factor;
    float ratio=pow(2.f, evolution_[0].octave);
    int sigmaSize=fRound(pointSize/ratio);
    border[0]=fRound(descriptorMax*sigmaSize);

    findExtrema(openclContext_, commandQueue_, evolution_[0].det, extremaMap_, evolution_[0].width, evolution_[0].height, 0, evolutionInfo_, threashold, options_.derivative_factor, border[0], keypointsCountBuffer, nullptr, event);
    events[0]=event;

    for(size_t i=1; i<evolution_.size(); i++)
    {
        pointSize=evolution_[i].sigma*options_.derivative_factor;
        ratio=pow(2.f, evolution_[i].octave);
        sigmaSize=fRound(pointSize/ratio);
        border[i]=fRound(descriptorMax*sigmaSize);

        findExtrema(openclContext_, commandQueue_, evolution_[i].det, extremaMap_, evolution_[i].width, evolution_[i].height, i, evolutionInfo_, threashold, options_.derivative_factor, border[i], keypointsCountBuffer, &events, event);

        events[0]=event;
    }

    keypointsCount_=0;
    commandQueue_.enqueueReadBuffer(keypointsCountBuffer, CL_FALSE, 0, sizeof(cl_int), &keypointsCount_, &events, &event);
    event.wait();

    if(keypointsCount_<=0)
        return;

    ::cl::Buffer keypointsTempBuffer=::cl::Buffer(openclContext_, CL_MEM_READ_WRITE, keypointsCount_*sizeof(cl::Keypoint));
    keypointsBuffer_=::cl::Buffer(openclContext_, CL_MEM_READ_WRITE, keypointsCount_*sizeof(cl::Keypoint));

    consolidateKeypoints(openclContext_, commandQueue_, extremaMap_, evolution_[0].width, evolution_[0].height, 0, evolutionInfo_, border[0], keypointsTempBuffer, keypointsCount_, keypointsCountBuffer, nullptr, event);
    events[0]=event;
    for(size_t i=1; i<evolution_.size(); i++)
    {
        consolidateKeypoints(openclContext_, commandQueue_, extremaMap_, evolution_[i].width, evolution_[i].height, i, evolutionInfo_, border[i], keypointsTempBuffer, keypointsCount_, keypointsCountBuffer, &events, event);
        events[0]=event;
    }
    timing_.extrema=timer.elapsedMs();

    subPixelRefinement(openclContext_, commandQueue_, evolutionDet_, evolutionInfo_, keypointsTempBuffer, keypointsCount_, keypointsBuffer_, keypointsCountBuffer, &events, event);
    event.wait();
    
    timing_.subpixel=timer.elapsedMs();
}

void AKAZE::Compute_Descriptors(std::vector<libAKAZE::Keypoint> &kpts, Descriptors &desc)
{
    putKeypoints(kpts);
    Compute_Descriptors(desc);
}

void AKAZE::Compute_Descriptors(Descriptors &desc)
{
    size_t descriptorSize;
    timer::Timer timer;

    //alocate space for descriptors
    if(options_.descriptor < MLDB_UPRIGHT)
        desc.float_descriptor.resize(keypointsCount_);
    else
    {
        // We use the full length binary descriptor -> 486 bits
        if(options_.descriptor_size==0)
            descriptorSize=(6+36+120) * options_.descriptor_channels;
        else
            descriptorSize=options_.descriptor_size;

        descriptorSize=ceil((float)descriptorSize/8);
        desc.binaryResize(keypointsCount_, descriptorSize);

        //allocate opencl uffer
        descriptorBufferSize_=descriptorSize*keypointsCount_;

        descriptorsBuffer_=::cl::Buffer(openclContext_, CL_MEM_READ_WRITE, descriptorBufferSize_);
    }


    switch(options_.descriptor)
    {
    case SURF_UPRIGHT:  // Upright descriptors, not invariant to rotation
        break;
    case SURF:
        break;
    case MSURF_UPRIGHT:  // Upright descriptors, not invariant to rotation
        break;
    case MSURF:
        break;
    case MLDB_UPRIGHT:
        break;
    case MLDB:
        {
            Compute_Main_Orientation();
            Get_MLDB_Full_Descriptor();

            std::vector<unsigned char> descriptorBuffer(descriptorBufferSize_);
            ::cl::Event event;

            commandQueue_.enqueueReadBuffer(descriptorsBuffer_, CL_FALSE, 0, descriptorBuffer.size(), descriptorBuffer.data(), nullptr, &event);
            event.wait();

            size_t index=0;
            for(int i=0; i<(int)(keypointsCount_); i++)
            {
                memcpy(desc.binaryData(), &descriptorBuffer[index], descriptorSize);
                index+=descriptorSize;
            }
        }
        break;
    }

    timing_.descriptor=timer.elapsedMs();
}

void AKAZE::Compute_Main_Orientation()
{
    ::cl::Event event;
    timer::Timer timer;

    computeOrientation(openclContext_, commandQueue_, keypointsBuffer_, keypointsCount_, evolutionDx_, evolutionDy_, evolutionInfo_, nullptr, event);
    
    event.wait();
    timing_.orientation=timer.elapsedMs();
}

void AKAZE::Get_MLDB_Full_Descriptor()
{
    ::cl::Event event;
    timer::Timer timer;

    getMLDBDescriptors(openclContext_, commandQueue_, keypointsBuffer_, descriptorsBuffer_, keypointsCount_, evolutionImage_, evolutionDx_, evolutionDy_, evolutionInfo_,
        options_.descriptor_channels, options_.descriptor_pattern_size, nullptr, event);

    event.wait();
    timing_.descriptoronly=timer.elapsedMs();
}


void AKAZE::getKeypoints(std::vector<libAKAZE::Keypoint> &kpts)
{
    std::vector<cl::Keypoint> keypoints(keypointsCount_);
    kpts.resize(keypointsCount_);

    ::cl::Event event;

    commandQueue_.enqueueReadBuffer(keypointsBuffer_, CL_FALSE, 0, keypoints.size()*sizeof(cl::Keypoint), keypoints.data(), nullptr, &event);
    event.wait();

    for(size_t i=0; i<keypoints.size(); ++i)
    {
        libAKAZE::Keypoint &keypoint=kpts[i];
        cl::Keypoint &clKeypoint=keypoints[i];

        keypoint.class_id=clKeypoint.class_id;
        keypoint.response=clKeypoint.response;
        keypoint.pt.x()=clKeypoint.ptX;
        keypoint.pt.y()=clKeypoint.ptY;
        keypoint.octave=clKeypoint.octave;
        keypoint.size=clKeypoint.size;
        keypoint.angle=clKeypoint.angle;
#ifdef TRACK_REMOVED
        keypoint.removed=clKeypoint.removed;
#endif //TRACK_REMOVED
    }
}

void AKAZE::putKeypoints(std::vector<libAKAZE::Keypoint> &kpts)
{
    keypointsCount_=kpts.size();
    std::vector<cl::Keypoint> keypoints(keypointsCount_);

    for(size_t i=0; i<kpts.size(); ++i)
    {
        libAKAZE::Keypoint &keypoint=kpts[i];
        cl::Keypoint &clKeypoint=keypoints[i];

        clKeypoint.class_id=keypoint.class_id;
        clKeypoint.response=keypoint.response;
        clKeypoint.ptX=keypoint.pt.x();
        clKeypoint.ptY=keypoint.pt.y();
        clKeypoint.octave=keypoint.octave;
        clKeypoint.size=keypoint.size;
        clKeypoint.angle=keypoint.angle;
    }

    keypointsBuffer_=::cl::Buffer(openclContext_, CL_MEM_READ_WRITE, keypoints.size()*sizeof(cl::Keypoint));

    ::cl::Event event;

    commandQueue_.enqueueWriteBuffer(keypointsBuffer_, CL_FALSE, 0, keypoints.size()*sizeof(cl::Keypoint), keypoints.data(), nullptr, &event);
    event.wait();
}

void AKAZE::Load_Keypoints(std::string fileName)
{
    FILE *file=fopen(fileName.c_str(), "rb");
    uint64_t count;

    fread(&count, sizeof(uint64_t), 1, file);

    std::vector<cl::Keypoint> keypoints(count);

    size_t size=fread(keypoints.data(), sizeof(cl::Keypoint), keypoints.size(), file);

    if(size != keypoints.size())
        throw std::runtime_error("Size read does not match expected size");

    fclose(file);
    
    keypointsBuffer_=::cl::Buffer(openclContext_, CL_MEM_READ_WRITE, keypoints.size()*sizeof(cl::Keypoint));
    keypointsCount_=keypoints.size();

    ::cl::Event event;

    commandQueue_.enqueueWriteBuffer(keypointsBuffer_, CL_FALSE, 0, keypoints.size()*sizeof(cl::Keypoint), keypoints.data(), nullptr, &event);
    event.wait();
}

void AKAZE::Save_Keypoints(std::string fileName)
{
    std::vector<cl::Keypoint> keypoints(keypointsCount_);
    ::cl::Event event;

    commandQueue_.enqueueReadBuffer(keypointsBuffer_, CL_FALSE, 0, keypoints.size()*sizeof(cl::Keypoint), keypoints.data(), nullptr, &event);
    
    FILE *file=fopen(fileName.c_str(), "wb");

    uint64_t count=keypointsCount_;

    fwrite(&count, sizeof(uint64_t), 1, file);

    event.wait();
    fwrite(keypoints.data(), sizeof(cl::Keypoint), keypoints.size(), file);
    fclose(file);
}

void AKAZE::Show_Computation_Times() const
{
    std::cout<<"(*) Time Scale Space: "<<timing_.scale<<std::endl;
    std::cout<<"   - Time kconstrast: "<<timing_.kcontrast<<std::endl;
    std::cout<<"   - Time Evolution: "<<timing_.evolution<<std::endl;
    std::cout<<"(*) Time Detector: "<<timing_.detector<<std::endl;
    std::cout<<"   - Time Derivatives: "<<timing_.derivatives<<std::endl;
    std::cout<<"   - Time Extrema: "<<timing_.extrema<<std::endl;
    std::cout<<"   - Time Subpixel: "<<timing_.subpixel<<std::endl;
    std::cout<<"(*) Time Descriptor: "<<timing_.descriptor<<std::endl;
    std::cout<<"   - Time Orientation: "<<timing_.orientation<<std::endl;
    std::cout<<"   - Time Descriptor Only: "<<timing_.descriptoronly<<std::endl;
    std::cout<<std::endl;
}

}}//namespace libAKAZE::cl