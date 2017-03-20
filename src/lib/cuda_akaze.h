//#include "AKAZEConfig.h"
#include "cudaImage.h"
#include <vector>
#include "akazeTypes.h"

float *AllocBuffers(int width, int height, int num, int omax, int &maxpts, std::vector<CudaImage> &buffers, libAKAZE::Keypoint *&pts,
    libAKAZE::Keypoint *&ptsbuffer, int *&ptindices, unsigned char *&desc, float *&descbuffer, CudaImage *&ims);
void InitCompareIndices();
void FreeBuffers(float *buffers);
double LowPass(CudaImage &inimg, CudaImage &outimg, CudaImage &temp, double var, int kernsize);
double Scharr(CudaImage &img, CudaImage &lx, CudaImage &ly);
double Flow(CudaImage &img, CudaImage &flow, libAKAZE::DIFFUSIVITY_TYPE type, float kcontrast);
double NLDStep(CudaImage &img,CudaImage &flow, CudaImage &temp, float stepsize);
double HalfSample(CudaImage &inimg, CudaImage &outimg);
double Copy(CudaImage &inimg, CudaImage &outimg);
double ContrastPercentile(CudaImage &img, CudaImage &temp, CudaImage &blur, float perc, int nbins, float &contrast);
double HessianDeterminant(CudaImage &img, CudaImage &lx, CudaImage &ly, int step);
double FindExtrema(CudaImage &img, CudaImage &imgp, CudaImage &imgn, float border, float dthreshold, int scale, int octave, float size, libAKAZE::Keypoint *pts, int maxpts);
void FilterExtrema(libAKAZE::Keypoint *pts, libAKAZE::Keypoint *newpts, int *kptindices, int &nump);
void ClearPoints();
int GetPoints(std::vector<libAKAZE::Keypoint>& h_pts, libAKAZE::Keypoint *d_pts, int numPts);
void WaitCuda();
void GetDescriptors(libAKAZE::Descriptors &h_desc, libAKAZE::Descriptors &d_desc, int numPts);
double FindOrientation(libAKAZE::Keypoint *d_pts, std::vector<CudaImage> &h_imgs, CudaImage *d_imgs, int numPts);
double ExtractDescriptors(libAKAZE::Keypoint *d_pts, std::vector<CudaImage> &cuda_buffers, CudaImage *cuda_images, unsigned char* desc_h, float *vals_d, int patsize, int numPts);
void MatchDescriptors(libAKAZE::Descriptors &desc_query, libAKAZE::Descriptors &desc_train,
		      std::vector<std::vector<libAKAZE::Match> > &dmatches,
		      size_t pitch, 
		      unsigned char* descq_d, unsigned char* desct_d, libAKAZE::Match *dmatches_d, libAKAZE::Match *dmatches_h);
void MatchDescriptors(libAKAZE::Descriptors &desc_query, libAKAZE::Descriptors &desc_train, std::vector<std::vector<libAKAZE::Match> > &dmatches);


struct CudaDevice
{
    int id;
    std::string name;
};

std::vector<CudaDevice> getCudaDevices();
