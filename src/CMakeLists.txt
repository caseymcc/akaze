# Find external libraries and dependencies
#hunter_add_package(OpenCV)
#find_package(OpenCV REQUIRED)

hunter_add_package(Eigen)
find_package(Eigen3 REQUIRED)

hunter_add_package(Jpeg)
find_package(JPEG CONFIG REQUIRED)

hunter_add_package(gflags)
find_package(gflags CONFIG REQUIRED)

find_package(Doxygen)
find_package(CUDA)

include(GenerateExportHeader)
include(CreateLaunchers)

option(AKAZE_USE_OPENCV "Add support for OpenCV" OFF)
option(AKAZE_USE_OPENCL "Add support for OpenCL" OFF)
option(AKAZE_USE_CUDA "Add support for CUDA" OFF)
option(AKAZE_USE_JSON "Add support for json" OFF)

if(AKAZE_USE_OPENCL)
	hunter_add_package(OpenCL)
	find_package(OpenCL CONFIG REQUIRED)
endif()

if(AKAZE_USE_JSON)
	hunter_add_package(RapidJSON)
	find_package(RapidJSON CONFIG REQUIRED)
endif()

add_subdirectory(cimg)
add_subdirectory(timer)
include_directories(${cimg_INCLUDE_DIR} ${timer_INCLUDE_DIR} ./)

set(AKAZE_USE_CUDA FALSE)
if(CUDA_FOUND)
  set(AKAZE_USE_CUDA TRUE)
endif()

# ===================================
# Set this to yes for python bindings
set(USE_PYTHON YES)


if(USE_PYTHON)
  find_package(Boost COMPONENTS python)
  find_package(PythonLibs)
  if (Boost_python_FOUND AND PYTHONLIBS_FOUND)
    include_directories(${PYTHON_INCLUDE_DIRS})
    include_directories(${Boost_INCLUDE_DIR})
  else()
    set(USE_PYTHON NO)
  endif()
endif(USE_PYTHON)

if (NOT APPLE)
  find_package(OpenMP)
endif ()

# ============================================================================ #
# Compilation flags
IF(UNIX)
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -g -O0  -Wall -Wextra -Wunused-variable -DDEBUG -D_DEBUG")
  SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g  -Wall -Wextra -Wunused-variable -DDEBUG -D_DEBUG")
  LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_50;-lineinfo;--compiler-options;-O2")
ENDIF(UNIX)

if(OPENMP_FOUND)
  MESSAGE("OpenMP found")
  if(UNIX)
    SET(CMAKE_C_FLAGS_RELEASE "-O3  -Wall -Wextra -Wunused-variable  -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
    SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wextra -Wunused-variable -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
  endif(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  else(OPENMP_FOUND)
    MESSAGE("OpenMP not found")
    if(UNIX)
      if(APPLE)
	SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3 -Wall -std=c++11 -Wunused-variable -Wno-unknown-pragmas -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
	SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -Wall -std=c++11 -stdlib=libc++ -Wno-unknown-pragmas -Wunused-variable -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
      else()
	SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3 -Wall -std=c++0x -Wunused-variable -Wno-unknown-pragmas -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
	SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -Wall -std=c++0x -Wno-unknown-pragmas -Wunused-variable -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
      endif()
    endif(UNIX)
endif(OPENMP_FOUND)


# ============================================================================ #
# Create projects
include_directories(${OpenCV_DIR})
include_directories("lib/" ${CUDASIFT})
include_directories(${CMAKE_CURRENT_LIST_DIR}/lib/include)

set(AKAZE_Runtime_Directories "")

# AKAZE headers
set(AKAZE_HEADERS
    lib/AKAZEConfig.h
    lib/AKAZE.h
	lib/akazeTypes.h
	lib/convolution.h
    lib/fed.h
    lib/nldiffusion_functions.h
    lib/utils.h
)

# AKAZE source
set(AKAZE_SRCS
    lib/AKAZE.cpp
	lib/convolution.cpp
    lib/fed.cpp
    lib/nldiffusion_functions.cpp
    lib/utils.cpp
)

if(AKAZE_USE_OPENCV)
	# AKAZE libraries
	set(AKAZE_LIBRARIES 
		${AKAZE_LIBRARIES}
		${OpenCV_LIBS}
	)

	SET(AKAZE_Runtime_Directories
		${_OpenCV_LIB_PATH}
	)

	#opencv's path has \'s
	string(REGEX REPLACE "\\\\" "/" AKAZE_Runtime_Directories ${AKAZE_Runtime_Directories}) 
endif()

if(AKAZE_USE_OPENCL)
    set(AKAZE_HEADERS ${AKAZE_HEADERS}
      lib/AKAZE_cl.h
	  lib/convolve_cl.h
      lib/filters_cl.h
      lib/openClContext.h
    )
    set(AKAZE_SRCS ${AKAZE_SRCS}
      lib/AKAZE_cl.cpp
	  lib/convolve_cl.cpp
      lib/filters_cl.cpp
      lib/openClContext.cpp
    )
	set(AKAZE_KERNELS
	  lib/kernels/contrast.cl
      lib/kernels/convolve.cl
    )

    set(AKAZE_LIBRARIES
        ${AKAZE_LIBRARIES}
        OpenCL::OpenCL
    )

	source_group("Kernels" FILES ${AKAZE_KERNELS})
	set_source_files_properties(${AKAZE_KERNELS} PROPERTIES HEADER_FILE_ONLY TRUE)

	set(AKAZE_HEADERS ${AKAZE_HEADERS} ${AKAZE_KERNELS})

    get_property(OpenCL_Runtime_Directory TARGET OpenCL::OpenCL PROPERTY LOCATION)
    get_filename_component(OpenCL_Runtime_Directory ${OpenCL_Runtime_Directory} DIRECTORY)

    set(AKAZE_Runtime_Directories ${AKAZE_Runtime_Directories} ${OpenCL_Runtime_Directory})
endif()

if(AKAZE_USE_JSON)
	set(AKAZE_LIBRARIES
        ${AKAZE_LIBRARIES}
		RapidJSON::rapidjson
    )
endif()

if(AKAZE_USE_CUDA)
    set(AKAZE_HEADERS ${AKAZE_HEADERS}
      lib/AKAZE_cuda.h
    )
    set(AKAZE_SRCS ${AKAZE_SRCS}
      lib/AKAZE_cuda.cpp
    )
endif()

configure_file(lib/config.h.in "${CMAKE_CURRENT_BINARY_DIR}/akaze_config.h" @ONLY)
set(AKAZE_HEADERS ${AKAZE_HEADERS} "${CMAKE_CURRENT_BINARY_DIR}/akaze_config.h")
if (USE_PYTHON)
  target_link_libraries(AKAZE ${Boost_PYTHON_LIBRARY} ${PYTHON_LIBRARIES})
endif()

add_definitions( -Dcimg_use_jpeg )

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_library(AKAZE ${AKAZE_HEADERS} ${AKAZE_SRCS})
#generating and export header that will end up in the binary directory
generate_export_header(AKAZE)

target_link_libraries(AKAZE Eigen3::Eigen JPEG::jpeg)
set(AKAZE_LIBRARIES ${AKAZE_LIBRARIES} timer)
target_link_libraries(AKAZE ${AKAZE_LIBRARIES})
set(AKAZE_LIBRARIES ${AKAZE_LIBRARIES} JPEG::jpeg)

if(AKAZE_USE_CUDA)
  SET(AKAZE_CUDA_SRCS
    lib/cudaImage.cu lib/cuda_akaze.cu)

  SET(CUDA_SEPARABLE_COMPILATION ON)

  #cuda code uses __shfl_down which requires compute_30
  list(APPEND CUDA_NVCC_FLAGS "--gpu-architecture=compute_30")

  get_property(EIGEN_INCLUDE_DIRS TARGET Eigen3::Eigen PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
  cuda_include_directories(${EIGEN_INCLUDE_DIRS})
  cuda_add_library(AKAZE_CUDA ${AKAZE_CUDA_SRCS} STATIC)

  target_link_libraries(AKAZE_CUDA ${OpenCV_LIBS} Eigen3::Eigen)

if(NOT BUILD_SHARED_LIBS)
  #for static builds both AKAZE and AKAZE_CUDA need to be added as libs, for shared AKAZE_CUDA should be part of AKAZE lib
  set(AKAZE_LIBRARIES ${AKAZE_LIBRARIES} AKAZE_CUDA) 
endif()

  target_link_libraries(AKAZE AKAZE_CUDA)
endif()

#get_property(AKAZE_Runtime_Directory TARGET AKAZE PROPERTY LOCATION_${CMAKE_BUILD_TYPE})
#get_filename_component(AKAZE_Runtime_Directory ${AKAZE_Runtime_Directory} DIRECTORY)
#set(AKAZE_Runtime_Directories ${AKAZE_Runtime_Directories} ${AKAZE_Runtime_Directory})
#message(FATAL_ERROR "AKAZE runtime:${AKAZE_Runtime_Directories}")

# Feature detector program
add_executable(akaze_features akaze_features.cpp commandLineHelper.cpp commandLineHelper.h)
target_link_libraries(akaze_features ${AKAZE_LIBRARIES} AKAZE)
create_target_launcher(akaze_features
	RUNTIME_LIBRARY_DIRS ${AKAZE_Runtime_Directories} 
)

add_executable(benchmark benchmark.cpp benchmark_opencl.h benchmark_opencl.cpp)
target_link_libraries(benchmark ${AKAZE_LIBRARIES} AKAZE gflags-static)
create_target_launcher(benchmark
	RUNTIME_LIBRARY_DIRS ${AKAZE_Runtime_Directories} 
)

# # Image matching program
# add_executable(akaze_match akaze_match.cpp)
# target_link_libraries(akaze_match ${AKAZE_LIBRARIES})
# 
# # Program that compares AKAZE with ORB and BRISK
# add_executable(akaze_compare akaze_compare.cpp)
# target_link_libraries(akaze_compare ${AKAZE_LIBRARIES})

# ============================================================================ #
# Library installation
set(AKAZE_INSTALL_HEADERS ${AKAZE_HEADERS})
install(TARGETS AKAZE DESTINATION lib)

install(FILES
    ${AKAZE_HEADERS}
    DESTINATION
    include
)

# ============================================================================ #
# Generate Doxygen-based documentation project
if(DOXYGEN_FOUND)
  add_custom_target(akaze_documentation
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)
