//=============================================================================
//
// AKAZE.cpp
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
* @file AKAZE_cuda.cpp
* @brief Main class for detecting and describing binary features in an
* accelerated nonlinear scale space
* @date Oct 07, 2014
* @author Pablo F. Alcantarilla, Jesus Nuevo
*/

#include "AKAZE_cuda.h"

#ifdef AKAZE_USE_CUDA

//#include <opencv2/highgui/highgui.hpp>
#include <cstdio>  //%%%%

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "timer/timer.hpp"

using namespace std;


namespace libAKAZE
{
namespace cuda
{

/* ************************************************************************* */
void Matcher::bfmatch(Descriptors &desc_query, Descriptors &desc_train,
    std::vector<std::vector<Match> > &dmatches)
{

    if(maxnquery < desc_query.size())
    {
        if(descq_d) cudaFree(descq_d);
        if(dmatches_d) cudaFree(dmatches_d);
        cudaMallocPitch((void**)&descq_d, &pitch, 64, desc_query.size());
        cudaMemset2D(descq_d, pitch, 0, 64, desc_query.size());
        cudaMalloc((void**)&dmatches_d, desc_query.size()*2*sizeof(Match));
        if(dmatches_h) delete[] dmatches_h;
        dmatches_h=new Match[2*desc_query.size()];
        maxnquery=desc_query.size();
    }
    if(maxntrain < desc_train.size())
    {
        if(desct_d) cudaFree(descq_d);
        cudaMallocPitch((void**)&desct_d, &pitch, 64, desc_train.size());
        cudaMemset2DAsync(desct_d, pitch, 0, 64, desc_train.size());
        maxntrain=desc_train.size();
    }

    cudaMemcpy2DAsync(descq_d, pitch, desc_query.binaryData(), desc_query.descriptorSize(),
        desc_query.descriptorSize(), desc_query.size(), cudaMemcpyHostToDevice);

    cudaMemcpy2DAsync(desct_d, pitch, desc_train.binaryData(), desc_train.descriptorSize(),
        desc_train.descriptorSize(), desc_train.size(), cudaMemcpyHostToDevice);

    dim3 block(desc_query.size());

    MatchDescriptors(desc_query, desc_train, dmatches, pitch,
        descq_d, desct_d, dmatches_d, dmatches_h);

    cudaMemcpy(dmatches_h, dmatches_d, desc_query.size()*2*sizeof(Match),
        cudaMemcpyDeviceToHost);

    dmatches.clear();
    for(int i=0; i < desc_query.size(); ++i)
    {
        std::vector<Match> tdmatch;
        //std::cout << dmatches_h[2*i].trainIdx << " - " << dmatches_h[2*i].queryIdx << std::endl;
        tdmatch.push_back(dmatches_h[2*i]);
        tdmatch.push_back(dmatches_h[2*i+1]);
        dmatches.push_back(tdmatch);
    }

}


//cv::Mat Matcher::bfmatch_(cv::Mat desc_query, cv::Mat desc_train)
//{
//
//    std::vector<std::vector<cv::DMatch> > dmatches_vec;
//
//    bfmatch(desc_query, desc_train, dmatches_vec);
//
//    cv::Mat dmatches_mat(dmatches_vec.size(), 8, CV_32FC1);
//
//    for(int i=0; i<dmatches_vec.size(); ++i)
//    {
//        float* mdata=(float*)&dmatches_mat.data[i*8*sizeof(float)];
//
//        mdata[0]=dmatches_vec[i][0].queryIdx;
//        mdata[1]=dmatches_vec[i][0].trainIdx;
//        mdata[2]=0.f;//dmatches_vec[i][0].imgIdx;
//        mdata[3]=dmatches_vec[i][0].distance;
//
//        mdata[4]=dmatches_vec[i][1].queryIdx;
//        mdata[5]=dmatches_vec[i][1].trainIdx;
//        mdata[6]=0.f;//dmatches_vec[i][1].imgIdx;
//        mdata[7]=dmatches_vec[i][1].distance;
//    }
//
//    return dmatches_mat;
//}


Matcher::~Matcher()
{
    if(descq_d)
    {
        cudaFree(descq_d);
    }
    if(desct_d)
    {
        cudaFree(desct_d);
    }
    if(dmatches_d)
    {
        cudaFree(dmatches_d);
    }
    if(dmatches_h)
    {
        delete[] dmatches_h;
    }
}



/* ************************************************************************* */
AKAZE::AKAZE(const Options& options): options_(options)
{
    ncycles_=0;
    reordering_=true;

    if(options_.descriptor_size > 0&&options_.descriptor>=MLDB_UPRIGHT)
    {
        generateDescriptorSubsample(
            descriptorSamples_, descriptorBits_, options_.descriptor_size,
            options_.descriptor_pattern_size, options_.descriptor_channels);
    }

    Allocate_Memory_Evolution();
}

/* ************************************************************************* */
AKAZE::~AKAZE()
{
    evolution_.clear();
    FreeBuffers(cuda_memory);
}

/* ************************************************************************* */
void AKAZE::Allocate_Memory_Evolution()
{
    generateEvolution(options_, evolution_);

    //Allocate memory for evolutions
    for(int i=0; i<evolution_.size(); ++i)
    {
        TEvolution &step=evolution_[i];

        step.Lx.resize(step.height, step.width);
        step.Ly.resize(step.height, step.width);
        step.Lxx.resize(step.height, step.width);
        step.Lxy.resize(step.height, step.width);
        step.Lyy.resize(step.height, step.width);
        step.Lt.resize(step.height, step.width);
        step.Ldet.resize(step.height, step.width);
        step.Lflow.resize(step.height, step.width);
        step.Lstep=Eigen::MatrixXf::Constant(step.height, step.width, 0.0); //.resize(step.height, step.width);
    }

    // Allocate memory for the number of cycles and time steps
    for(size_t i=1; i < evolution_.size(); i++)
    {
        int naux=0;
        vector<float> tau;
        float ttime=0.0;
        ttime=evolution_[i].time-evolution_[i-1].time;
        float tmax=0.25;// * (1 << 2 * evolution_[i].octave);
        naux=fed_tau_by_process_time(ttime, 1, tmax, reordering_, tau);
        nsteps_.push_back(naux);
        tsteps_.push_back(tau);
        ncycles_++;
    }

    // Allocate memory for CUDA buffers
//    options_.ncudaimages=4*options_.nsublevels;
    ncudaimages=4*options_.nsublevels;
    maxkeypoints=16*8192;

    unsigned char* _cuda_desc;
    cuda_memory=AllocBuffers(
        evolution_[0].Lt.cols(), evolution_[0].Lt.rows(), ncudaimages,
        options_.omax, maxkeypoints, cuda_buffers, cuda_bufferpoints,
        cuda_points, cuda_ptindices, _cuda_desc, cuda_descbuffer, cuda_images);
    
    cuda_desc.binaryResize(maxkeypoints, 61);
}

/* ************************************************************************* */
int AKAZE::Create_Nonlinear_Scale_Space(const RowMatrixXf& img)
{
    if(evolution_.size()==0)
    {
        cerr<<"Error generating the nonlinear scale space!!"<<endl;
        cerr<<"Firstly you need to call AKAZE::Allocate_Memory_Evolution()"
            <<endl;
        return -1;
    }

    timer::Timer timer;

    TEvolution& ev=evolution_[0];
    CudaImage& Limg=cuda_buffers[0];
    CudaImage& Lt=cuda_buffers[0];
    CudaImage& Lsmooth=cuda_buffers[1];
    CudaImage& Ltemp=cuda_buffers[2];

    Limg.h_data=(float*)img.data();
    Limg.Download();

    ContrastPercentile(Limg, Ltemp, Lsmooth, options_.kcontrast_percentile,
        options_.kcontrast_nbins, options_.kcontrast);
    LowPass(Limg, Lt, Ltemp, options_.soffset * options_.soffset,
        2*ceil((options_.soffset-0.8)/0.3)+3);
    Copy(Lt, Lsmooth);

    Lt.h_data=(float*)ev.Lt.data();

    timing_.kcontrast=timer.elapsedMs();

    // Now generate the rest of evolution levels
    for(size_t i=1; i < evolution_.size(); i++)
    {
        TEvolution& evn=evolution_[i];
        int num=ncudaimages;
        CudaImage& Lt=cuda_buffers[evn.octave * num+0+4*evn.sublevel];
        CudaImage& Lsmooth=cuda_buffers[evn.octave * num+1+4*evn.sublevel];
        CudaImage& Lstep=cuda_buffers[evn.octave * num+2];
        CudaImage& Lflow=cuda_buffers[evn.octave * num+3];

        TEvolution& evo=evolution_[i-1];
        CudaImage& Ltold=cuda_buffers[evo.octave * num+0+4*evo.sublevel];
        if(evn.octave > evo.octave)
        {
            HalfSample(Ltold, Lt);
            options_.kcontrast=options_.kcontrast * 0.75;
        }
        else
            Copy(Ltold, Lt);

        LowPass(Lt, Lsmooth, Lstep, 1.0, 5);
        Flow(Lsmooth, Lflow, options_.diffusivity, options_.kcontrast);

        for(int j=0; j < nsteps_[i-1]; j++)
        {
            float stepsize=tsteps_[i-1][j]/(1<<2*evn.octave);
            // NLDStep(Lt, Lflow, Lstep, stepsize);
            NLDStep(Lt, Lflow, Lstep, tsteps_[i-1][j]);
        }

        Lt.h_data=(float*)evn.Lt.data();
    }

    timing_.evolution=timer.elapsedMs();

    return 0;
}


//void kpvec2mat(std::vector<cv::KeyPoint>& kpts, cv::Mat& _mat)
//{
//
//    _mat=cv::Mat(kpts.size(), 7, CV_32FC1);
//    for(int i=0; i<(int)kpts.size(); ++i)
//    {
//
//
//    }
//
//}
//
//
//void mat2kpvec(cv::Mat& _mat, std::vector<cv::KeyPoint>& _kpts)
//{
//
//    for(int i=0; i<_mat.rows; ++i)
//    {
//        cv::Vec<float, 7> v=_mat.at<cv::Vec<float, 7> >(i, 0);
//        cv::KeyPoint kp(v[0], v[1], v[2], v[3], v[4], (int)v[5], (int)v[6]);
//        _kpts.push_back(kp);
//    }
//
//}



//cv::Mat AKAZE::Feature_Detection_()
//{
//
//    std::vector<cv::KeyPoint> kpts;
//
//    this->Feature_Detection(kpts);
//
//    cv::Mat mat;
//    kpvec2mat(kpts, mat);
//
//    return mat;
//}


/* ************************************************************************* */
void AKAZE::Feature_Detection(std::vector<Keypoint>& kpts)
{
    
    timer::Timer timer;

    int num=ncudaimages;
    for(size_t i=0; i < evolution_.size(); i++)
    {
        TEvolution& ev=evolution_[i];
        CudaImage& Lsmooth=cuda_buffers[ev.octave * num+1+4*ev.sublevel];
        CudaImage& Lx=cuda_buffers[ev.octave * num+2+4*ev.sublevel];
        CudaImage& Ly=cuda_buffers[ev.octave * num+3+4*ev.sublevel];

        float ratio=pow(2.0f, (float)evolution_[i].octave);
        int sigma_size_=
            fRound(evolution_[i].sigma * options_.derivative_factor/ratio);
        HessianDeterminant(Lsmooth, Lx, Ly, sigma_size_);

        Lx.h_data=(float*)evolution_[i].Lx.data();
        Ly.h_data=(float*)evolution_[i].Ly.data();
    }
    timing_.derivatives=timer.elapsedMs();

    ClearPoints();
    for(size_t i=0; i < evolution_.size(); i++)
    {
        TEvolution& ev=evolution_[i];
        TEvolution& evp=evolution_[(
            i > 0&&evolution_[i].octave==evolution_[i-1].octave ? i-1 : i)];
        TEvolution& evn=
            evolution_[(i < evolution_.size()-1&&
                evolution_[i].octave==evolution_[i+1].octave
                ? i+1
                : i)];
        CudaImage& Ldet=cuda_buffers[ev.octave * num+1+4*ev.sublevel];
        CudaImage& LdetP=cuda_buffers[evp.octave * num+1+4*evp.sublevel];
        CudaImage& LdetN=cuda_buffers[evn.octave * num+1+4*evn.sublevel];

        float smax=1.0f;
        if(options_.descriptor==SURF_UPRIGHT||options_.descriptor==SURF||
            options_.descriptor==MLDB_UPRIGHT||options_.descriptor==MLDB)
            smax=10.0 * sqrtf(2.0f);
        else if(options_.descriptor==MSURF_UPRIGHT||
            options_.descriptor==MSURF)
            smax=12.0 * sqrtf(2.0f);

        float ratio=pow(2.0f, (float)evolution_[i].octave);
        float size=evolution_[i].sigma * options_.derivative_factor;
        float border=smax * fRound(size/ratio);
        float thresh=std::max(options_.dthreshold, options_.min_dthreshold);

        FindExtrema(Ldet, LdetP, LdetN, border, thresh, i, evolution_[i].octave,
            size, cuda_points, maxkeypoints);
    }

    FilterExtrema(cuda_points, cuda_bufferpoints, cuda_ptindices, nump);

    //GetPoints(kpts, cuda_points);


    timing_.extrema=timer.elapsedMs();
}


#ifdef USE_PYTHON
boost::python::tuple AKAZE::Compute_Descriptors_()
{

    std::vector<cv::KeyPoint> kptsvec;

    this->Feature_Detection(kptsvec);

    cv::Mat desc;
    cv::Mat kpts;
    this->Compute_Descriptors(kptsvec, desc);

    kpvec2mat(kptsvec, kpts);

    return boost::python::make_tuple(desc, kpts);

}
#endif // USE_PYTHON


/* ************************************************************************* */
/**
* @brief This method  computes the set of descriptors through the nonlinear
* scale space
* @param kpts Vector of detected keypoints
* @param desc Matrix to store the descriptors
*/
void AKAZE::Compute_Descriptors(std::vector<Keypoint>& kpts,
    Descriptors &desc)
{
    timer::Timer timer;
    size_t descriptorSize;

    // Allocate memory for the matrix with the descriptors
    if(options_.descriptor < MLDB_UPRIGHT)
    {
        descriptorSize=64;
        desc.floatResize(kpts.size(), descriptorSize);
    }
    else
    {
        int t;

        if(options_.descriptor_size==0)
            t=(6+36+120) * options_.descriptor_channels;
        else
            t=options_.descriptor_size;

        descriptorSize=ceil((float)t/8);
        desc.binaryResize(kpts.size(), descriptorSize);
    }

    int pattern_size=options_.descriptor_pattern_size;

    switch(options_.descriptor)
    {
    case MLDB:
        {
            timer::Timer timer;

            FindOrientation(cuda_points, cuda_buffers, cuda_images, nump);
            timing_.orientation=timer.elapsedMs();

            GetPoints(kpts, cuda_points, nump);
            ExtractDescriptors(cuda_points, cuda_buffers, cuda_images,
                cuda_desc.binaryData(), cuda_descbuffer, pattern_size, nump);
            GetDescriptors(desc, cuda_desc, nump);
            timing_.descriptoronly=timer.elapsedMs();
        }
        break;
    case SURF_UPRIGHT:
    case SURF:
    case MSURF_UPRIGHT:
    case MSURF:
    case MLDB_UPRIGHT:
        cout<<"Descriptor not implemented\n";
    }

    timing_.descriptor=timer.elapsedMs();

    WaitCuda();
}


/* ************************************************************************* */
void AKAZE::Save_Scale_Space()
{
//    cv::Mat img_aux;
//    string outputFile;
//    // TODO Readback and save
//    for(size_t i=0; i < evolution_.size(); i++)
//    {
//        convert_scale(evolution_[i].Lt);
//        evolution_[i].Lt.convertTo(img_aux, CV_8U, 255.0, 0);
//        outputFile="../output/evolution_"+to_formatted_string(i, 2)+".jpg";
//        cv::imwrite(outputFile, img_aux);
//    }
}

/* ************************************************************************* */
void AKAZE::Save_Detector_Responses()
{
//    cv::Mat img_aux;
//    string outputFile;
//    float ttime=0.0;
//    int nimgs=0;
//
//    for(size_t i=0; i < evolution_.size(); i++)
//    {
//        ttime=evolution_[i+1].etime-evolution_[i].etime;
//        if(ttime > 0)
//        {
//            convert_scale(evolution_[i].Ldet);
//            evolution_[i].Ldet.convertTo(img_aux, CV_8U, 255.0, 0);
//            outputFile=
//                "../output/images/detector_"+to_formatted_string(nimgs, 2)+".jpg";
//            imwrite(outputFile.c_str(), img_aux);
//            nimgs++;
//        }
//    }
}

/* ************************************************************************* */
void AKAZE::Show_Computation_Times() const
{
    std::cout<<"(*) Time Scale Space: "<<timing_.scale<<std::endl;
    std::cout<<"   - Time kcontrast: "<<timing_.kcontrast<<std::endl;
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



/* ************************************************************************* */
void check_descriptor_limits(int& x, int& y, int width,
    int height)
{
    if(x < 0) x=0;

    if(y < 0) y=0;

    if(x > width-1) x=width-1;

    if(y > height-1) y=height-1;
}

std::vector<CudaDevice> getDevices()
{
    std::vector<::CudaDevice> cudaDevices=getCudaDevices();

    std::vector<CudaDevice> devices;

    for(::CudaDevice &cudaDevice:cudaDevices)
    {
        CudaDevice device;

        device.id=cudaDevice.id;
        device.name=cudaDevice.name;

        devices.push_back(device);
    }
    return devices;
}

}}//namespace libAKAZE::cuda

#endif //AKAZE_USE_CUDA