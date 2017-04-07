#ifndef _filters_cl_h_
#define _filters_cl_h_

#include "CL/cl.hpp"
#include "Eigen/Core"
#include "AKAZEConfig.h"
#include "convolve_cl.h"


namespace libAKAZE{namespace cl
{
///
///
///
void calculateMagnitude(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &lx, ::cl::Image2D &ly, ::cl::Image2D &magnitude, size_t width, size_t height, std::vector<::cl::Event> *events, ::cl::Event &event);

///
///
///
float calculateMax(::cl::Context &context, ::cl::CommandQueue &commandQueue, ::cl::Image2D &src, size_t width, size_t height, std::vector<::cl::Event> *events, ::cl::Event &event);
void calculateMax(::cl::Context &context, ::cl::CommandQueue &commandQueue, ::cl::Image2D &src, size_t width, size_t height, ::cl::Buffer &maxBuffer, std::vector<::cl::Event> *events, ::cl::Event &event);

///
///
///
std::vector<int> calculateHistogram(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, size_t width, size_t height, int bins, float scale, std::vector<::cl::Event> *events, ::cl::Event &event);
//void calculateHistogram(::cl::Context &context, ::cl::CommandQueue &commandQueue, ::cl::Image2D &src, size_t width, size_t height, int bins, ::cl::Buffer maxValueBuffer,
//    ::cl::Buffer &histogramBuffer, ::cl::Buffer &scratchBuffer, std::vector<::cl::Event> *events, ::cl::Event &event);
void calculateHistogram(::cl::Context &context, ::cl::CommandQueue &commandQueue, ::cl::Image2D &src, size_t width, size_t height, int bins, float percent, ::cl::Buffer maxValueBuffer,
    ::cl::Buffer &histogramBuffer, ::cl::Buffer contrastBuffer, ::cl::Buffer &scratchBuffer, std::vector<::cl::Event> *events, ::cl::Event &event);

///
///
///
float computeKPercentile(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, float perc, float gscale, size_t nbins);
void computeKPercentile(::cl::Context &context, ::cl::CommandQueue &commandQueue, ::cl::Image2D &src, int width, int height, size_t nbins, float percent,
    ::cl::Image2D &gaussian, ::cl::Image2D &magnitude, ::cl::Buffer &histogramBuffer, ::cl::Buffer &histogramScratchBuffer,
    ::cl::Buffer &guassianKernel, int guassiankernelSize, ::cl::Image2D &scratch, ::cl::Image2D lx, ::cl::Image2D ly, ScharrSeparableKernel &scharrKernel, int scharrKernelSize,
    ::cl::Buffer &maxValue, ::cl::Buffer &contrast, std::vector<::cl::Event> *events, ::cl::Event &event);
float computeContrast(std::vector<int> &histogram, float hmax, float perc);

///
///
///
void linearSample(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src, ::cl::Image2D &dst, size_t dstWidth, size_t dstHeight, std::vector<::cl::Event> *events, ::cl::Event &event);

///
///
///
void pmG1(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src1, ::cl::Image2D &src2, ::cl::Image2D &dst, size_t width, size_t height, float contrast, std::vector<::cl::Event> *events, ::cl::Event &event);
//void pmG2(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src1, ::cl::Image2D &src2, ::cl::Image2D &dst, size_t width, size_t height, float contrast, std::vector<::cl::Event> *events, ::cl::Event &event);
void pmG2(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src1, ::cl::Image2D &src2, ::cl::Image2D &dst, size_t width, size_t height, int index, ::cl::Buffer &contrastBuffer, std::vector<::cl::Event> *events, ::cl::Event &event);
void weickert(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src1, ::cl::Image2D &src2, ::cl::Image2D &dst, size_t width, size_t height, float contrast, std::vector<::cl::Event> *events, ::cl::Event &event);
void charbonnier(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &src1, ::cl::Image2D &src2, ::cl::Image2D &dst, size_t width, size_t height, float contrast, std::vector<::cl::Event> *events, ::cl::Event &event);

///
///
///
void nldStepScalar(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &image, ::cl::Image2D &flow, ::cl::Image2D &step, size_t width, size_t height, float stepsize, std::vector<::cl::Event> *events, ::cl::Event &event);

///
///
///
void scale(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &image, size_t width, size_t height, float scale, std::vector<::cl::Event> *events, ::cl::Event &event);

///
///
///
void determinantHessian(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &dxx, ::cl::Image2D &dyy, ::cl::Image2D &dxy, size_t width, size_t height, ::cl::Buffer &output, std::vector<::cl::Event> *events, ::cl::Event &event);
void determinantHessian(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Buffer &dxx, size_t dxxOffset, ::cl::Buffer &dyy, size_t dyyOffset, ::cl::Buffer &dxy, size_t dxyOffset,
    size_t width, size_t height, ::cl::Buffer &output, size_t outputOffset, std::vector<::cl::Event> *events, ::cl::Event &event);


///
///
///
void findExtrema(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &detImage, size_t width, size_t height, ::cl::Buffer &featureMap, size_t featureMapWidth, size_t featureMapHeight, 
    int evolutionClass, int octave, float sigma, float threshold, float derivativeFactor, int border, std::vector<::cl::Event> *events, ::cl::Event &event);
void findExtremaIterate(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Image2D &detImage, size_t width, size_t height, ::cl::Buffer &featureMap, size_t featureMapWidth, size_t featureMapHeight,
    int evolutionClass, int octave, float sigma, float threshold, float derivativeFactor, int border, std::vector<::cl::Event> *events, ::cl::Event &event);

void findExtrema(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Buffer &detImage, ::cl::Buffer &featureMap, int width, int height, int mapIndex, ::cl::Buffer evolutionBuffer, float threshold, float derivativeFactor, int border,
    ::cl::Buffer keypointCount, std::vector<::cl::Event> *events, ::cl::Event &event);
void consolidateKeypoints(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Buffer &featureMap, int width, int height, int mapIndex, ::cl::Buffer evolutionBuffer, int border,
    ::cl::Buffer keypoints, int maxKeypoints, ::cl::Buffer count, std::vector<::cl::Event> *events, ::cl::Event &event);
void subPixelRefinement(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Buffer &det, ::cl::Buffer evolutionBuffer, ::cl::Buffer keypoints, int keypointCount,
    ::cl::Buffer filteredKeypoints, ::cl::Buffer keypointCountBuffer, std::vector<::cl::Event> *events, ::cl::Event &event);

//void findExtrema(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Buffer &detImage, size_t offset, size_t width, size_t height, ::cl::Buffer &featureMap, size_t featureMapWidth, size_t featureMapHeight,
//    int evolutionClass, int octave, float sigma, float threshold, float derivativeFactor, int border, std::vector<::cl::Event> *events, ::cl::Event &event);
//void findExtremaIterate(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Buffer &detImage, size_t offset, size_t width, size_t height, ::cl::Buffer &featureMap, size_t featureMapWidth, size_t featureMapHeight,
//    int evolutionClass, int octave, float sigma, float threshold, float derivativeFactor, int border, std::vector<::cl::Event> *events, ::cl::Event &event);
//void filterExtrema(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Buffer &featureMap, size_t featureMapWidth, size_t featureMapHeight, int border,
//    ::cl::Buffer keypointCount, std::vector<::cl::Event> *events, ::cl::Event &event);
//
//void consolidateKeypoints(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Buffer &featureMap, size_t featureMapWidth, size_t featureMapHeight, int border,
//    ::cl::Buffer keypoints, int maxKeypoints, ::cl::Buffer count, std::vector<::cl::Event> *events, ::cl::Event &event);

void computeOrientation(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Buffer &keypoints, size_t count, ::cl::Buffer &dx, ::cl::Buffer &dy,
    ::cl::Buffer &evolutionInfo, std::vector<::cl::Event> *events, ::cl::Event &event);


void getMLDBDescriptors(::cl::Context context, ::cl::CommandQueue commandQueue, ::cl::Buffer &keypoints, ::cl::Buffer descriptors, size_t count, ::cl::Buffer &image, ::cl::Buffer &dx, ::cl::Buffer &dy,
    ::cl::Buffer &evolutionInfo, int channels, int patternSize, std::vector<::cl::Event> *events, ::cl::Event &event);
///
/// Convolves to matrices
///
template <typename Derived1_, typename Derived2_>
Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> convolveMatrix(const Eigen::MatrixBase<Derived1_>& matrix1, const Eigen::MatrixBase<Derived2_> &matrix2)
{
    size_t cols=matrix1.cols()+matrix2.cols()-1;
    size_t rows=matrix1.rows()+matrix2.rows()-1;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> output=Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Zero(rows, cols);
    
    size_t matrix1Row=0;
    size_t matrix2Row=matrix2.rows()-1;
    size_t rowSize=1;

    for(size_t y=0; y<rows; ++y)
    {
        size_t matrix1Col=0;
        size_t matrix2Col=matrix2.cols()-1;
        size_t colSize=1;

        for(size_t x=0; x<cols; ++x)
        {
            Derived1_::Scalar b=(matrix1.block(matrix1Row, matrix1Col, rowSize, colSize).cwiseProduct(matrix2.block(matrix2Row, matrix2Col, rowSize, colSize))).sum();
            output.coeffRef(y, x)=b;

            if((matrix1Col <= matrix1.cols()-1) && (matrix2Col>=0))
            {
                if((colSize<matrix1.cols())&&(colSize<matrix2.cols()))
                    colSize++;
                matrix2Col--;
            }
            else
            {
                matrix1Col++;
                if(colSize > matrix1.cols()-matrix1Col)
                    colSize--;
            }
        }

        if((matrix1Row<=matrix1.rows()-1) && (matrix2Row>0))
        {
            if((rowSize<matrix1.rows())&&(rowSize<matrix2.rows()))
                rowSize++;
            matrix2Row--;
        }
        else
        {
            matrix1Row++;
            if(rowSize > matrix1.rows()-matrix1Row)
                rowSize--;
        }
    }

    return output;
}

}}//namespace libAKAZE::cl

#endif //_filters_cl_h_