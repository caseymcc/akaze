#include "commandLineHelper.h"

#include <iostream>

using namespace std;

/* ************************************************************************* */
int parse_input_options(ProgramOptions &options, libAKAZE::Options &akaze_options, int argc, char* argv[])
{
    bool ignoreOutput=false;

    // If there is only one argument return
    if(argc==1)
    {
//        show_input_options_help(0);
        return -2;
    }
    // Set the options from the command line
    else if(argc>=2)
    {
        akaze_options=libAKAZE::Options();
        options.keypoints_path="./keypoints.txt";

        if(!strcmp(argv[1], "--help"))
        {
//            show_input_options_help(0);
            return -2;
        }

        options.image_path=argv[1];

        for(int i=2; i<argc; i++)
        {
            if(!strcmp(argv[i], "--type"))
            {
                i=i+1;
                if(i>=argc)
                {
                    cerr<<"Error selecting processing type!!"<<endl;
                    return -1;
                }
                else
                {
                    if(strcmp(argv[i], "Cuda")==0)
                        akaze_options.type=libAKAZE::Options::Cuda;
                    else if(strcmp(argv[i], "OpenCL")==0)
                        akaze_options.type=libAKAZE::Options::OpenCL;
                    else
                        akaze_options.type=libAKAZE::Options::Standard;
                }
            }
            else if(!strcmp(argv[i], "--devices"))
            {
                options.requestDevices=true;
            }
            else if(!strcmp(argv[i], "--platform"))
            {
                i=i+1;
                if(i>=argc)
                {
                    cerr<<"Error selecting processing platform!!"<<endl;
                    return -1;
                }
                else
                {
                    options.platform=argv[i];
                }
            }
            else if(!strcmp(argv[i], "--device"))
            {
                i=i+1;
                if(i>=argc)
                {
                    cerr<<"Error selecting processing device!!"<<endl;
                    return -1;
                }
                else
                {
                    options.device=argv[i];
                }
            }
            else if(!strcmp(argv[i], "--soffset"))
            {
                i=i+1;
                if(i>=argc)
                {
                    cerr<<"Error introducing input options!!"<<endl;
                    return -1;
                }
                else
                {
                    akaze_options.soffset=atof(argv[i]);
                }
            }
            else if(!strcmp(argv[i], "--omax"))
            {
                i=i+1;
                if(i>=argc)
                {
                    cerr<<"Error introducing input options!!"<<endl;
                    return -1;
                }
                else
                {
                    akaze_options.omax=atof(argv[i]);
                }
            }
            else if(!strcmp(argv[i], "--dthreshold"))
            {
                i=i+1;
                if(i>=argc)
                {
                    cerr<<"Error introducing input options!!"<<endl;
                    return -1;
                }
                else
                {
                    akaze_options.dthreshold=atof(argv[i]);
                }
            }
            else if(!strcmp(argv[i], "--sderivatives"))
            {
                i=i+1;
                if(i>=argc)
                {
                    cerr<<"Error introducing input options!!"<<endl;
                    return -1;
                }
                else
                {
                    akaze_options.sderivatives=atof(argv[i]);
                }
            }
            else if(!strcmp(argv[i], "--nsublevels"))
            {
                i=i+1;
                if(i>=argc)
                {
                    cerr<<"Error introducing input options!!"<<endl;
                    return -1;
                }
                else
                    akaze_options.nsublevels=atoi(argv[i]);
            }
            else if(!strcmp(argv[i], "--diffusivity"))
            {
                i=i+1;
                if(i>=argc)
                {
                    cerr<<"Error introducing input options!!"<<endl;
                    return -1;
                }
                else
                    akaze_options.diffusivity=libAKAZE::DIFFUSIVITY_TYPE(atoi(argv[i]));
            }
            else if(!strcmp(argv[i], "--descriptor"))
            {
                i=i+1;
                if(i>=argc)
                {
                    cerr<<"Error introducing input options!!"<<endl;
                    return -1;
                }
                else
                {
                    akaze_options.descriptor=libAKAZE::DESCRIPTOR_TYPE(atoi(argv[i]));

                    if(akaze_options.descriptor < 0||akaze_options.descriptor > libAKAZE::MLDB)
                    {
                        akaze_options.descriptor=libAKAZE::MLDB;
                    }
                }
            }
            else if(!strcmp(argv[i], "--descriptor_channels"))
            {
                i=i+1;
                if(i>=argc)
                {
                    cerr<<"Error introducing input options!!"<<endl;
                    return -1;
                }
                else
                {
                    akaze_options.descriptor_channels=atoi(argv[i]);

                    if(akaze_options.descriptor_channels<=0||
                        akaze_options.descriptor_channels>3)
                    {
                        akaze_options.descriptor_channels=3;
                    }
                }
            }
            else if(!strcmp(argv[i], "--descriptor_size"))
            {
                i=i+1;
                if(i>=argc)
                {
                    cerr<<"Error introducing input options!!"<<endl;
                    return -1;
                }
                else
                {
                    akaze_options.descriptor_size=atoi(argv[i]);

                    if(akaze_options.descriptor_size<0)
                    {
                        akaze_options.descriptor_size=0;
                    }
                }
            }
            else if(!strcmp(argv[i], "--load_scale_space"))
            {
                i=i+1;
                if(i>=argc)
                {
                    cerr<<"Error introducing input options!!"<<endl;
                    return -1;
                }
                else
                {
                    options.save_scale_space=false;
                    options.scale_space_directory=argv[i];
                }
            }
            else if(!strcmp(argv[i], "--save_scale_space"))
            {
                i=i+1;
                if(i>=argc)
                {
                    cerr<<"Error introducing input options!!"<<endl;
                    return -1;
                }
                else
                {
                    options.save_scale_space=true;
                    options.scale_space_directory=argv[i];
                }
            }
            else if(!strcmp(argv[i], "--load_det_hessian_response"))
            {
                i=i+1;
                if(i>=argc)
                {
                    cerr<<"Error introducing input options!!"<<endl;
                    return -1;
                }
                else
                {
                    options.save_det_hessian_response=false;
                    options.det_hessian_response_directory=argv[i];
                }
            }
            else if(!strcmp(argv[i], "--save_det_hessian_response"))
            {
                i=i+1;
                if(i>=argc)
                {
                    cerr<<"Error introducing input options!!"<<endl;
                    return -1;
                }
                else
                {
                    options.save_det_hessian_response=true;
                    options.det_hessian_response_directory=argv[i];
                }
            }
            else if(!strcmp(argv[i], "--show_results"))
            {
                i=i+1;
                if(i>=argc)
                {
                    cerr<<"Error introducing input options!!"<<endl;
                    return -1;
                }
                else
                {
                    options.show_results=(bool)atoi(argv[i]);
                }
            }
            else if(!strcmp(argv[i], "--verbose"))
            {
                akaze_options.verbosity=true;
            }
            else if(!strcmp(argv[i], "--load_keypoints"))
            {
                i=i+1;
                if(i>=argc)
                {
                    cerr<<"Error introducing input options!!"<<endl;
                    return -1;
                }
                else
                {
                    options.save_keypoints=false;
                    options.keypoints_path=argv[i];
                    ignoreOutput=true;
                }
            }
            else if(!strcmp(argv[i], "--save_keypoints"))
            {
                i=i+1;
                if(i>=argc)
                {
                    cerr<<"Error introducing input options!!"<<endl;
                    return -1;
                }
                else
                {
                    options.save_keypoints=true;
                    options.keypoints_data_path=argv[i];
                    ignoreOutput=true;
                }
            }
            else if(!strcmp(argv[i], "--output") && ! ignoreOutput)
            {
                i=i+1;
                if(i>=argc)
                {
                    cerr<<"Error introducing input options!!"<<endl;
                    return -1;
                }
                else
                    options.keypoints_path=argv[i];
            }
        }
    }

    return 0;
}