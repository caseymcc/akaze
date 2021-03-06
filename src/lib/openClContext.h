#ifndef _openCL_Context_h_
#define _openCL_Context_h_

#include <CL/cl.hpp>
#include "akaze_export.h"
#include <limits>

namespace libAKAZE
{
namespace cl
{

class OpenCLContext;

OpenCLContext *getOpenClContext(::cl::Context context);
::cl::Kernel getKernel(::cl::Context context, std::string kernelName, std::string fileName);

struct AKAZE_EXPORT OpenClDevice
{
    enum Type
    {
        CPU,
        GPU
    };
    cl_device_id deviceId;
    std::string name;
    Type type;

    std::string platform;
    std::string vendor;
    std::string version;
};

AKAZE_EXPORT std::vector<OpenClDevice> getDevices();

AKAZE_EXPORT ::cl::Context openDevice();
AKAZE_EXPORT ::cl::Context openDevice(std::string deviceName);
AKAZE_EXPORT ::cl::Context openDevice(std::string platform, std::string deviceName);

void saveBufferAsImage(::cl::CommandQueue commandQueue, ::cl::Buffer &clBuffer, size_t offset, size_t width, size_t height, std::string fileName, float minClip=std::numeric_limits<float>::lowest(), float maxClip=std::numeric_limits<float>::max());
void loadBufferFromImage(::cl::CommandQueue commandQueue, ::cl::Buffer &clBuffer, size_t offset, size_t width, size_t height, std::string fileName);

void saveImage2D(::cl::CommandQueue commandQueue, ::cl::Image2D &clImage, std::string fileName);
void loadImage2D(::cl::CommandQueue commandQueue, ::cl::Image2D &clImage, std::string fileName);

void saveImage2DData(::cl::CommandQueue commandQueue, ::cl::Image2D &clImage, std::string fileName);
void loadImage2DData(::cl::CommandQueue commandQueue, ::cl::Image2D &clImage, std::string fileName);
void loadImage2DData(::cl::CommandQueue commandQueue, ::cl::Image2D &clImage, std::string fileName, ::cl::Event &event);

void saveImage2DCsv(::cl::CommandQueue commandQueue, ::cl::Image2D &clImage, std::string fileName);

void saveBufferCsv(::cl::CommandQueue commandQueue, ::cl::Buffer &clBuffer, size_t width, size_t height, size_t offset, std::string fileName);

void loadBufferData(::cl::CommandQueue commandQueue, ::cl::Buffer &clBuffer, size_t offset, float *buffer, size_t bufferSize, std::string fileName, ::cl::Event *event);
void loadBufferData(::cl::CommandQueue commandQueue, ::cl::Buffer &clBuffer, size_t offset, std::string fileName);

void saveBufferData(::cl::CommandQueue commandQueue, ::cl::Buffer &clBuffer, size_t offset, size_t width, size_t height, std::string fileName);

}}//namespace libAKAZE::cl

#endif //_openCL_Context_h_