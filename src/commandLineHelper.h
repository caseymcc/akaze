#include "AKAZEConfig.h"

struct ProgramOptions
{
    std::string image_path;

    bool save_keypoints;
    std::string keypoints_data_path;

    std::string keypoints_path;

    bool show_results;
    bool save_scale_space;
    std::string scale_space_directory;
    bool save_det_hessian_response;
    std::string det_hessian_response_directory;

#ifdef AKAZE_USE_OPENCL
    bool requestDevices;

    std::string platform;
    std::string device;
#endif //AKAZE_USE_OPENCL

    ProgramOptions()
    {
        save_keypoints=false;
        show_results=false;
        save_scale_space=false;
        save_det_hessian_response=false;
#ifdef AKAZE_USE_OPENCL
        requestDevices=false;
#endif //AKAZE_USE_OPENCL
    }
};

int parse_input_options(ProgramOptions &options, libAKAZE::Options &akaze_options, int argc, char* argv[]);
