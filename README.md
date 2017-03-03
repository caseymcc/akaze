## README - A-KAZE Features

This is a fork of libAKAZE with the intention of consolidating the following projects and adding support for OpenCL.

- <https://github.com/pablofdezalc/akaze>  - original
- <https://github.com/h2suzuki/fast_akaze>  - faster CPU implmentation
- https://github.com/sweeneychris/akaze-eigen  - removal of OpenCV, uses eigen for API
- https://github.com/nbergst/akaze - CUDA implementation


The interface is as close to possible as the original version. For the CPU version it is the same. Namespaces have been introduced for both the CUDA (libAKAZE::cuda) and OpenCL (libAKAZE::cl) versions.

For a detailed description, please refer to <https://github.com/pablofdezalc/akaze> and the above project pages.


## Current status
- Eigen has replaced OpenCV in API
- FastAkaze changes have not been implemented
- OpenCL version (rough implementation almost complete)
 - Feature point extraction complete
 - Descriptor extraction almost complete
 - Planning a brute force matcher
- CUDA version (currently disabled)
 - Implementation moved to libAKAZE::cuda
 - Functionality current not tested
 - Need to modify API to work off eigen

## Benchmarks
The following benchmarks are measured on the img1.pgm in the iguazu dataset provided by the original authors, and are averages over 100 runs. The computer is a 16 core Xeon running at 2.6 GHz with 32 GB of RAM and an Nvidia Titan X (Maxwell). The operating system is Ubuntu 14.04, with CUDA 8.0.

| Operation     | CPU (original) (ms)      | CUDA (ms) | OpenCL (ms) |
| ------------- |:------------------------:|:---------:|:-----------:|
| Detection     |            117           |    6.5    |     TBD     |
| Descriptor    |            10            |    0.9    |     TBD     |


## CUDA

Just changing namespace from libAKAZE to libAKAZECU should be enough. Keypoints and descriptors are returned on the CPU for later matching etc. using e.g. OpenCV. We also provide a rudimentary brute force matcher running on the GPU.

This code was created as a joint effort between
- Niklas Bergström https://github.com/nbergst
- Mårten Björkman https://github.com/Celebrandil
- Alessandro Pieropan https://github.com/CoffeRobot

#### Optimizations (from https://github.com/nbergst/akaze)
The code has been optimized with the goal to maintain the same interface as well as to produce the same results as the original code. This means that certain tradeoffs have been necessary, in particular in the way keypoints are filtered. One difference remains though related to finding scale space extrema, which has been reported as an issue here: <https://github.com/pablofdezalc/akaze/issues/24>
Major optimizations are possible, but this is work in progress. These optimizations will relax the constraint of having results that are identical to the original code.

#### Matcher
A not very optimized matcher is also provided. It returns a std::vector\<std::vector\<cv::DMatch\>\> with the two closest matches.

#### Limitations
- Previous limitations with respect to the number of keypoints are more or less gone. Set the maximum number of keypoints in AKAZEConfig.h. This is done since cuda memory is preallocated..
- The only descriptor available is MLDB, as proposed in the original authors' paper.
- Currently it only works with 4 octaves and 4 sub-levels (default settings).


## OpenCL 

This code was created as a joint effort between
- Casey McCandless https://github.com/caseymcc

#### Optimizations
Currently the port is almost a direct port of the code from akaze-eigen. Modification have only been made were appropriate to getting it functional in OpenCL. Optimization opportunities are abundant.


## Citation
If you use this code as part of your research, please cite the following papers:

CUDA version

1. **Feature Descriptors for Tracking by Detection: a Benchmark**. Alessandro Pieropan, Mårten Björkman, Niklas Bergström and Danica Kragic (arXiv:1607.06178).

Original A-KAZE papers

2. **Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces**. Pablo F. Alcantarilla, J. Nuevo and Adrien Bartoli. _In British Machine Vision Conference (BMVC), Bristol, UK, September 2013_

3. **KAZE Features**. Pablo F. Alcantarilla, Adrien Bartoli and Andrew J. Davison. _In European Conference on Computer Vision (ECCV), Fiorenze, Italy, October 2012_
