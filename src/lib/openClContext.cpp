#include "openCLContext.h"

#include <memory>
#include <unordered_map>
#include <cstdio>
#include <fstream>
#include <iomanip>

#include "cimg/CImg.h"

#include <memory>

namespace libAKAZE
{
namespace cl
{

void checkInfo(::cl::Device device);
void getDeviceInfo(::cl::Device device, OpenClDevice &deviceInfo);

typedef std::unordered_map<std::string, SharedKernelInfo> KernelMap;
typedef std::unordered_map<std::string, ::cl::Program> ProgramMap;

class OpenCLContext
{
public:
    OpenCLContext() {}
    ~OpenCLContext()
    {
        kernels.clear();
        programs.clear();
    }

//    std::vector<::cl::Device> devices;
    ::cl::Device device;
    ::cl::Context context;
    
    OpenClDevice deviceInfo;

    KernelMap kernels;
    ProgramMap programs;
};

typedef std::shared_ptr<OpenCLContext> SharedOpenCLContext;
typedef std::unordered_map<::cl::Context::cl_type, SharedOpenCLContext> OpenCLContexts;

OpenCLContext *getOpenClContext(::cl::Context context)
{
    static OpenCLContexts s_openCLContext;
    OpenCLContexts::iterator iter=s_openCLContext.find(context());

    if(iter!=s_openCLContext.end())
        return iter->second.get();

    SharedOpenCLContext openCLContext=std::make_shared<OpenCLContext>();

    openCLContext->context=context;

    std::vector<::cl::Device> devices;

    context.getInfo(CL_CONTEXT_DEVICES, &devices);

    //assuming only 1 device in use
    openCLContext->device=devices[0];
    getDeviceInfo(openCLContext->device, openCLContext->deviceInfo);

    s_openCLContext.insert({context(), openCLContext});

    return openCLContext.get();
}

::cl::Program getProgram(::cl::Context context, std::string fileName)
{
    OpenCLContext *openCLContext=getOpenClContext(context);

    if(openCLContext==nullptr)
        return ::cl::Program();

    ProgramMap &programs=openCLContext->programs;
    ProgramMap::iterator iter=programs.find(fileName);

    if(iter!=programs.end())
        return ::cl::Program(iter->second);

    //no kernel built, attempt to create it
    FILE *file=fopen(fileName.c_str(), "r");

    if(file==NULL)
        return ::cl::Program();

    size_t size;
    
    fseek(file, 0L, SEEK_END);
    size=ftell(file);
    rewind(file);
    
    std::string source;
    
    source.resize(size+1);
	size_t fileSize=fread((void *)source.data(), sizeof(char), size, file);
    fclose(file);
	
	source.resize(fileSize); 
    
    ::cl::Program::Sources programSource(1, std::make_pair(source.data(), source.size()));
    ::cl::Program program=::cl::Program(context, programSource);

    cl_int error;
    std::vector<::cl::Device> devices(1);
    
    devices[0]=openCLContext->device;

#if !defined(NDEBUG)
    std::string buildCommandline="-g";

#ifdef LINUX
    char path[PATH_MAX];

    realpath(fileName.c_str(), path);
    buildCommandline+=" -s ";
    buildCommandline+=path;
#endif
#if _WINDOWS
    char path[_MAX_PATH];

    _fullpath(path, fileName.c_str(), _MAX_PATH);
    buildCommandline+=" -s ";
    buildCommandline+=path;
#endif
//    error=program.build(devices, buildCommandline.c_str());
    error=program.build(devices, "");
#else
    error=program.build(devices);
#endif

    if(error!=CL_SUCCESS)
    {
        if(error==CL_BUILD_PROGRAM_FAILURE)
        {
            std::string str=program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(openCLContext->device);

            printf(" \n\t\t\tBUILD LOG\n");
            printf(" ************************************************\n");
            printf(str.c_str());
            printf(" ************************************************\n");

            return false;
        }
    }

    programs.insert({fileName, program});

    return program;
}

::cl::Kernel getKernel(::cl::Context context, std::string kernelName, std::string fileName)
{
    SharedKernelInfo kernelInfo=getKernelInfo(context, kernelName, fileName);

    if(kernelInfo)
        return kernelInfo->kernel;

    return ::cl::Kernel();
}

SharedKernelInfo getKernelInfo(::cl::Context context, std::string kernelName, std::string fileName)
{
    OpenCLContext *openCLContext=getOpenClContext(context);

    if(openCLContext==nullptr)
        return SharedKernelInfo();

    KernelMap &kernels=openCLContext->kernels;

    KernelMap::iterator iter=kernels.find(kernelName);

    if(iter!=kernels.end())
        return iter->second;

    ::cl::Program program=getProgram(context, fileName);

    SharedKernelInfo kernelInfo=std::make_shared<KernelInfo>();

    cl_int error;
    kernelInfo->kernel=::cl::Kernel(program, kernelName.c_str(), &error);

    kernelInfo->kernel.getWorkGroupInfo(openCLContext->device, CL_KERNEL_WORK_GROUP_SIZE, &kernelInfo->workGroupSize);
    kernelInfo->kernel.getWorkGroupInfo(openCLContext->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &kernelInfo->preferredWorkGroupMultiple);
    kernelInfo->kernel.getWorkGroupInfo(openCLContext->device, CL_KERNEL_LOCAL_MEM_SIZE, &kernelInfo->localMemoryUsed);

    //compying info so it is available when setting up kernel call
    kernelInfo->deviceComputeUnits=openCLContext->deviceInfo.computeUnits;
    kernelInfo->deviceLocalMemory=openCLContext->deviceInfo.localMemory;

    kernels.insert({kernelName, kernelInfo});

    return kernelInfo;
}

void getDeviceInfo(::cl::Device device, OpenClDevice &deviceInfo)
{
    std::string platformName;
    std::string vendor;
    std::string version;
    cl_platform_id platformId;
    cl_device_type deviceType;

    device.getInfo(CL_DEVICE_PLATFORM, &platformId);

    ::cl::Platform platform(platformId);

    platform.getInfo(CL_PLATFORM_NAME, &platformName);

    //platformName can be null-terminated
    if(platformName.back()==0)
        platformName.pop_back();

    //trim white space
    platformName.erase(platformName.find_last_not_of(" \n\r\t\0")+1);

    platform.getInfo(CL_PLATFORM_VENDOR, &vendor);
    platform.getInfo(CL_PLATFORM_VERSION, &version);

    deviceInfo.deviceId=device();

    deviceInfo.platform=platformName;
    deviceInfo.vendor=vendor;
    deviceInfo.version=version;

    device.getInfo(CL_DEVICE_TYPE, &deviceType);
    device.getInfo(CL_DEVICE_NAME, &deviceInfo.name);
    device.getInfo(CL_DEVICE_BUILT_IN_KERNELS, &deviceInfo.builtInKernels);
    device.getInfo(CL_DEVICE_EXTENSIONS, &deviceInfo.extensions);

    device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &deviceInfo.computeUnits);
    device.getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &deviceInfo.globalCache);
    device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &deviceInfo.globalMemory);
    device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &deviceInfo.localMemory);
    device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &deviceInfo.maxWorkGroupSize);
    device.getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &deviceInfo.maxWorkItemDims);
    device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &deviceInfo.maxWorkItemSizes);

    //device name can be null-terminated
    if(deviceInfo.name.back()==0)
        deviceInfo.name.pop_back();

    //trim white space
    deviceInfo.name.erase(deviceInfo.name.find_last_not_of(" \n\r\t\0")+1);

    switch(deviceType)
    {
    case CL_DEVICE_TYPE_CPU:
        deviceInfo.type=OpenClDevice::CPU;
        break;
    case CL_DEVICE_TYPE_GPU:
        deviceInfo.type=OpenClDevice::GPU;
        break;
    }
}

std::vector<OpenClDevice> getDevices()
{
    std::vector<::cl::Platform> platforms;
    ::cl::Platform::get(&platforms);
    std::vector<OpenClDevice> devices;

    for(size_t i=0; i<platforms.size(); ++i)
    {
//        cl_int error;
        std::string platformName;
        std::string vendor;
        std::string version;

        ::cl::Platform &platform=platforms[i];

        platform.getInfo(CL_PLATFORM_NAME, &platformName);

        //platformName can be null-terminated
        if(platformName.back()==0)
            platformName.pop_back();

        //trim white space
        platformName.erase(platformName.find_last_not_of(" \n\r\t\0")+1);

        platform.getInfo(CL_PLATFORM_VENDOR, &vendor);
        platform.getInfo(CL_PLATFORM_VERSION, &version);

        std::vector<::cl::Device> platformDevices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &platformDevices);

        for(::cl::Device &device:platformDevices)
        {
            OpenClDevice deviceInfo;
//            cl_device_type deviceType;
//
//            deviceInfo.deviceId=device();
//            
//            deviceInfo.platform=platformName;
//            deviceInfo.vendor=vendor;
//            deviceInfo.version=version;
//
//            device.getInfo(CL_DEVICE_TYPE, &deviceType);
//            device.getInfo(CL_DEVICE_NAME, &deviceInfo.name);
//
//            //device name can be null-terminated
//            if(deviceInfo.name.back()==0)
//                deviceInfo.name.pop_back();
//
//            //trim white space
//            deviceInfo.name.erase(deviceInfo.name.find_last_not_of(" \n\r\t\0")+1);
//
//            switch(deviceType)
//            {
//            case CL_DEVICE_TYPE_CPU:
//                deviceInfo.type=OpenClDevice::CPU;
//                break;
//            case CL_DEVICE_TYPE_GPU:
//                deviceInfo.type=OpenClDevice::GPU;
//                break;
//            }
            getDeviceInfo(device, deviceInfo);
            devices.push_back(deviceInfo);
        }
    }

    return devices;
}

::cl::Context openDevice(OpenClDevice &deviceInfo)
{
    //user has asked us to get a device, lets assume they wanted a GPU so lets try that first and if no device exists
    //then return first available CPU

    std::vector<::cl::Platform> platforms;
    ::cl::Platform::get(&platforms);

    for(size_t i=0; i<platforms.size(); ++i)
    {
        ::cl::Platform &platform=platforms[i];
        std::vector<::cl::Device> platformDevices;

        platform.getDevices(CL_DEVICE_TYPE_ALL, &platformDevices);
        
        for(::cl::Device &device:platformDevices)
        {
            cl_device_type deviceType;

            device.getInfo(CL_DEVICE_TYPE, &deviceType);

            if(deviceType!=CL_DEVICE_TYPE_GPU)
                continue;

            ::cl::Context context(device);

            getDeviceInfo(device, deviceInfo);

            checkInfo(device);
            return context;
        }

        //no longer picky take whatever
        for(::cl::Device &device:platformDevices)
        {
            ::cl::Context context(device);

            getDeviceInfo(device, deviceInfo);
            checkInfo(device);
            return context;
        }
    }

    return ::cl::Context();
}

::cl::Context openDevice(std::string deviceName, OpenClDevice &deviceInfo)
{
    std::vector<::cl::Platform> platforms;
    ::cl::Platform::get(&platforms);
    cl_device_id deviceId;
    bool found=false;

    for(size_t i=0; i<platforms.size(); ++i)
    {
        ::cl::Platform &platform=platforms[i];
        std::vector<::cl::Device> platformDevices;

        platform.getDevices(CL_DEVICE_TYPE_ALL, &platformDevices);
        for(::cl::Device &device:platformDevices)
        {
            std::string localDeviceName;

            device.getInfo(CL_DEVICE_NAME, &localDeviceName);

            //localDeviceName can be null-terminated
            if(localDeviceName.back()==0)
                localDeviceName.pop_back();

            //trim white space
            localDeviceName.erase(localDeviceName.find_last_not_of(" \n\r\t\0")+1);

            if(localDeviceName == deviceName)
            {
                deviceId=device();
                found=true;

                getDeviceInfo(device, deviceInfo);

                break;
            }
        }

        if(found)
            break;
    }

    if(!found)
        return ::cl::Context();

    ::cl::Device device(deviceId);
    ::cl::Context context(device);

    checkInfo(device);

    return context;
}

::cl::Context openDevice(std::string platformName, std::string deviceName, OpenClDevice &deviceInfo)
{
    std::vector<::cl::Platform> platforms;
    ::cl::Platform::get(&platforms);
    cl_device_id deviceId;
    bool found=false;

    for(size_t i=0; i<platforms.size(); ++i)
    {
        std::string localPlatformName;

        ::cl::Platform &platform=platforms[i];

        platform.getInfo(CL_PLATFORM_NAME, &localPlatformName);

        //localPlatformName can be null-terminated
        if(localPlatformName.back()==0)
            localPlatformName.pop_back();

        //trim white space
        localPlatformName.erase(localPlatformName.find_last_not_of(" \n\r\t\0")+1);

        if(localPlatformName != platformName)
            continue;

        std::vector<::cl::Device> platformDevices;

        platform.getDevices(CL_DEVICE_TYPE_ALL, &platformDevices);
        for(::cl::Device &device:platformDevices)
        {
            std::string localDeviceName;

            device.getInfo(CL_DEVICE_NAME, &localDeviceName);

            //localDeviceName can be null-terminated
            if(localDeviceName.back()==0)
                localDeviceName.pop_back();

            //trim white space
            localDeviceName.erase(localDeviceName.find_last_not_of(" \n\r\t\0")+1);

            if(localDeviceName == deviceName)
            {
                deviceId=device();
                found=true;
                
                getDeviceInfo(device, deviceInfo);
                
                break;
            }
        }

        if(found)
            break;
    }

    if(!found)
        return ::cl::Context();

    ::cl::Device device(deviceId);
    ::cl::Context context(device);

    checkInfo(device);

    return context;
}

void checkInfo(::cl::Device device)
{
    std::string builtInKernels;
    std::string extensions;
    cl_ulong globalCache;
    cl_ulong globalMemory;
    cl_ulong localMemory;
    size_t maxWorkGroupSize;
    cl_uint maxWorkItemDims;
    std::vector<size_t> maxWorkItemSizes;

    device.getInfo(CL_DEVICE_BUILT_IN_KERNELS, &builtInKernels);
    device.getInfo(CL_DEVICE_EXTENSIONS, &extensions);

    device.getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &globalCache);
    device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &globalMemory);
    device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &localMemory);
    device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);
    device.getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &maxWorkItemDims);
    device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &maxWorkItemSizes);
}

void saveBufferAsImage(::cl::CommandQueue commandQueue, ::cl::Buffer &clBuffer, size_t offset, size_t width, size_t height, std::string fileName, float minClip, float maxClip)
{
    std::vector<float> imageBuffer(width*height);
    cimg_library::CImg<float> image(imageBuffer.data(), width, height, 1, 1, true);

    size_t bufferSize;
    size_t imageSize=image._width*image._height*sizeof(cl_float);
    size_t offsetSize=offset*sizeof(cl_float);
    size_t minBufferSize=imageSize+offsetSize;

    clBuffer.getInfo(CL_MEM_SIZE, &bufferSize);

    if(minBufferSize>bufferSize)
    {
        throw std::out_of_range("Buffer size does not match expected Image size");
        return;
    }

    ::cl::Event event;

    commandQueue.enqueueReadBuffer(clBuffer, CL_FALSE, offsetSize, imageSize, imageBuffer.data(), nullptr, &event);
    event.wait();

    if(minClip > std::numeric_limits<float>::lowest())
    {
        for(size_t i=0; i<imageBuffer.size(); ++i)
        {
            if(imageBuffer[i]<minClip)
                imageBuffer[i]=minClip;
        }
    }

    if(maxClip < std::numeric_limits<float>::max())
    {
        for(size_t i=0; i<imageBuffer.size(); ++i)
        {
            if(imageBuffer[i]>maxClip)
                imageBuffer[i]=maxClip;
        }
    }

    image.normalize(0, 255);
    image.save(fileName.c_str());

}

void loadBufferFromImage(::cl::CommandQueue commandQueue, ::cl::Buffer &clBuffer, size_t offset, std::string fileName)
{
    cimg_library::CImg<float> image(fileName.c_str());

    size_t bufferSize;
    size_t imageSize=image._width*image._height*sizeof(cl_float);
    size_t offsetSize=offset*sizeof(cl_float);
    size_t minBufferSize=imageSize+offsetSize;

    clBuffer.getInfo(CL_MEM_SIZE, &bufferSize);

    if(minBufferSize>bufferSize)
    {
        throw std::out_of_range("Image size does not match expected Buffer size");
        return;
    }

    ::cl::Event event;

    commandQueue.enqueueWriteBuffer(clBuffer, CL_FALSE, offset, imageSize, image._data);
    event.wait();
}

void saveImage2D(::cl::CommandQueue commandQueue, ::cl::Image2D &clImage, std::string fileName)
{
    size_t width, height;
    cl_image_format format;

    clImage.getImageInfo(CL_IMAGE_WIDTH, &width);
    clImage.getImageInfo(CL_IMAGE_HEIGHT, &height);
//    clImage.getImageInfo(CL_IMAGE_FORMAT, &format);

    std::vector<float> imageBuffer(width*height);
    ::cl::size_t<3> origin;
    ::cl::size_t<3> region;

    origin[0]=0;
    origin[1]=0;
    origin[2]=0;

    region[0]=width;
    region[1]=height;
    region[2]=1;

    cimg_library::CImg<float> image(imageBuffer.data(), width, height, 1, 1, true);

    ::cl::Event event;

    commandQueue.enqueueReadImage(clImage, CL_FALSE, origin, region, 0, 0, imageBuffer.data(), nullptr, &event);
    event.wait();

    image.normalize(0, 255);
    image.save(fileName.c_str());
}

void loadImage2D(::cl::CommandQueue commandQueue, ::cl::Image2D &clImage, std::string fileName)
{
    size_t width, height;
    cl_image_format format;

    clImage.getImageInfo(CL_IMAGE_WIDTH, &width);
    clImage.getImageInfo(CL_IMAGE_HEIGHT, &height);
//    clImage.getImageInfo(CL_IMAGE_FORMAT, &format);

    cimg_library::CImg<float> image(fileName.c_str());

    if((width!=image._width)||(height=image._height))
    {
        throw std::out_of_range("OpenCL Image size does not match expected Image size");
        return;
    }

    ::cl::size_t<3> origin;
    ::cl::size_t<3> region;

    origin[0]=0;
    origin[1]=0;
    origin[2]=0;

    region[0]=width;
    region[1]=height;
    region[2]=1;

    ::cl::Event event;

    commandQueue.enqueueWriteImage(clImage, CL_FALSE, origin, region, 0, 0, image._data, nullptr, &event);
    event.wait();
}

void loadImage2DData(::cl::CommandQueue commandQueue, ::cl::Image2D &clImage, std::string fileName, ::cl::Event &event)
{
    uint64_t width, height;
    uint64_t format;

    FILE *file=fopen(fileName.c_str(), "rb");
    
    fread(&format, sizeof(uint64_t), 1, file);
    fread(&width, sizeof(uint64_t), 1, file);
    fread(&height, sizeof(uint64_t), 1, file);

    std::vector<float> imageBuffer(width*height);

    size_t size=fread(imageBuffer.data(), sizeof(float), imageBuffer.size(), file);

    if(size!=imageBuffer.size())
        throw std::runtime_error("Size read does not match expected size");

    ::cl::size_t<3> origin;
    ::cl::size_t<3> region;

    origin[0]=0;
    origin[1]=0;
    origin[2]=0;

    region[0]=width;
    region[1]=height;
    region[2]=1;

	//CL_TRUE needed as opencl needs to copy the buffer as it will go out of scope shortly
    commandQueue.enqueueWriteImage(clImage, CL_TRUE, origin, region, 0, 0, imageBuffer.data(), nullptr, &event);
}

void loadImage2DData(::cl::CommandQueue commandQueue, ::cl::Image2D &clImage, std::string fileName)
{
    ::cl::Event event;
    
    loadImage2DData(commandQueue, clImage, fileName, event);

    event.wait();
}

void saveImage2DData(::cl::CommandQueue commandQueue, ::cl::Image2D &clImage, std::string fileName)
{
    size_t width, height;
    cl_image_format format;

    clImage.getImageInfo(CL_IMAGE_WIDTH, &width);
    clImage.getImageInfo(CL_IMAGE_HEIGHT, &height);
//    clImage.getImageInfo(CL_IMAGE_FORMAT, &format);

    std::vector<float> imageBuffer(width*height);
    ::cl::size_t<3> origin;
    ::cl::size_t<3> region;

    origin[0]=0;
    origin[1]=0;
    origin[2]=0;

    region[0]=width;
    region[1]=height;
    region[2]=1;

    ::cl::Event event;

    commandQueue.enqueueReadImage(clImage, CL_FALSE, origin, region, 0, 0, imageBuffer.data(), nullptr, &event);
    event.wait();

    uint64_t oWidth=width;
    uint64_t oHeight=height;
    uint64_t oFormat=4;
    FILE *file=fopen(fileName.c_str(), "wb");

    fwrite(&oFormat, sizeof(int64_t), 1, file);
    fwrite(&oWidth, sizeof(int64_t), 1, file);
    fwrite(&oHeight, sizeof(int64_t), 1, file);

    size_t size=fwrite(imageBuffer.data(), sizeof(float), imageBuffer.size(), file);

    if(size!=imageBuffer.size())
        throw std::runtime_error("Size written does not match image size");

    fclose(file);
}

void saveImage2DCsv(::cl::CommandQueue commandQueue, ::cl::Image2D &clImage, std::string fileName)
{
	size_t width, height;
	cl_image_format format;

	clImage.getImageInfo(CL_IMAGE_WIDTH, &width);
	clImage.getImageInfo(CL_IMAGE_HEIGHT, &height);
	//    clImage.getImageInfo(CL_IMAGE_FORMAT, &format);

	std::vector<float> imageBuffer(width*height);
	::cl::size_t<3> origin;
	::cl::size_t<3> region;

	origin[0]=0;
	origin[1]=0;
	origin[2]=0;

	region[0]=width;
	region[1]=height;
	region[2]=1;

    ::cl::Event event;

	commandQueue.enqueueReadImage(clImage, CL_FALSE, origin, region, 0, 0, imageBuffer.data(), nullptr, &event);

	std::ofstream file(fileName);
	float value;
	
	file<<std::setprecision(4);

    event.wait();

	for(int y=0; y<height; y++)
	{
		int yPos=width*y;
		file<<imageBuffer[yPos];
		for(int x=1; x<width; x++)
		{
			file<<", "<<imageBuffer[yPos+x];
		}
		file<<"\n";
	}
}

void saveBufferCsv(::cl::CommandQueue commandQueue, ::cl::Buffer &clBuffer, size_t offset, size_t width, size_t height, std::string fileName)
{
    size_t bufferSize;
    size_t imageSize=width*height*sizeof(cl_float);
    size_t offsetSize=offset*sizeof(cl_float);
    size_t minBufferSize=imageSize+offsetSize;
    std::vector<float> imageBuffer(width*height);

    clBuffer.getInfo(CL_MEM_SIZE, &bufferSize);

    if(minBufferSize>bufferSize)
    {
        throw std::out_of_range("Buffer size does not match expected Image size");
        return;
    }

    ::cl::Event event;

    commandQueue.enqueueReadBuffer(clBuffer, CL_FALSE, offsetSize, imageSize, imageBuffer.data(), nullptr, &event);

    std::ofstream file(fileName);
    float value;

    file<<std::setprecision(4);
    event.wait();

    for(int y=0; y<height; y++)
    {
        int yPos=width*y;
        file<<imageBuffer[yPos];
        for(int x=1; x<width; x++)
        {
            file<<", "<<imageBuffer[yPos+x];
        }
        file<<"\n";
    }
}

void loadBufferData(::cl::CommandQueue commandQueue, ::cl::Buffer &clBuffer, size_t offset, float *buffer, size_t bufferSize, std::string fileName, ::cl::Event *event)
{
    uint64_t width, height;
    uint64_t format;
	size_t memSize;

    FILE *file=fopen(fileName.c_str(), "rb");

    fread(&format, sizeof(uint64_t), 1, file);
    fread(&width, sizeof(uint64_t), 1, file);
    fread(&height, sizeof(uint64_t), 1, file);

	clBuffer.getInfo(CL_MEM_SIZE, &memSize);

    size_t size=width*height*sizeof(float);
	size_t offsetSize=offset*sizeof(float);

    if(bufferSize < size)
        throw std::runtime_error("Buffer size cannot hold data");

	if (memSize < offsetSize+size)
		throw std::runtime_error("Buffer size cannot hold data");

    size_t readSize=fread(buffer, 1, size, file);

    if(readSize!=size)
        throw std::runtime_error("Size read does not match expected size");

    fclose(file);

    commandQueue.enqueueWriteBuffer(clBuffer, CL_TRUE, offsetSize, size, buffer, nullptr, event);
}

void loadBufferData(::cl::CommandQueue commandQueue, ::cl::Buffer &clBuffer, size_t offset, std::string fileName)
{
    uint64_t width, height;
    uint64_t format;
    size_t memSize;

    FILE *file=fopen(fileName.c_str(), "rb");

    fread(&format, sizeof(uint64_t), 1, file);
    fread(&width, sizeof(uint64_t), 1, file);
    fread(&height, sizeof(uint64_t), 1, file);

    clBuffer.getInfo(CL_MEM_SIZE, &memSize);

	size_t size=width*height*sizeof(float);
	size_t offsetSize=offset*sizeof(float);

    if(memSize < offsetSize+size)
        throw std::runtime_error("Buffer size cannot hold data");

    std::vector<float> imageBuffer(width*height);

    size_t readSize=fread(imageBuffer.data(), sizeof(float), imageBuffer.size(), file);

    if(readSize!=imageBuffer.size())
        throw std::runtime_error("Size read does not match expected size");

    fclose(file);

    ::cl::Event event;

    commandQueue.enqueueWriteBuffer(clBuffer, CL_FALSE, offsetSize, size, imageBuffer.data(), nullptr, &event);
    event.wait();
}

void saveBufferData(::cl::CommandQueue commandQueue, ::cl::Buffer &clBuffer, size_t offset, size_t width, size_t height, std::string fileName)
{
    std::vector<float> imageBuffer(width*height);
	size_t offsetSize = offset * sizeof(float);
    ::cl::Event event;

    commandQueue.enqueueReadBuffer(clBuffer, CL_FALSE, offsetSize, imageBuffer.size()*sizeof(float), imageBuffer.data(), nullptr, &event);

    uint64_t oWidth=width;
    uint64_t oHeight=height;
    uint64_t oFormat=4;
    FILE *file=fopen(fileName.c_str(), "wb");

    fwrite(&oFormat, sizeof(int64_t), 1, file);
    fwrite(&oWidth, sizeof(int64_t), 1, file);
    fwrite(&oHeight, sizeof(int64_t), 1, file);

    event.wait();

    fwrite(imageBuffer.data(), sizeof(float), imageBuffer.size(), file);
    fclose(file);
}

}}//namespace libAKAZE::cl