 # Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a
 # copy of this software and associated documentation files (the "Software"),
 # to deal in the Software without restriction, including without limitation
 # the rights to use, copy, modify, merge, publish, distribute, sublicense,
 # and/or sell copies of the Software, and to permit persons to whom the
 # Software is furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in
 # all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 # THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 # FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 # DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
# The version of CMake which is not compatible with the old CUDA CMake commands.
set(CMAKE_VERSION_THRESHOLD "3.10.0")

if(${CMAKE_VERSION} VERSION_LESS ${CMAKE_VERSION_THRESHOLD})
  project(onnx2trt LANGUAGES CXX C)
else()
  project(onnx2trt LANGUAGES CXX C CUDA)
endif() 


#
# CUDA Configuration
# This is no longer necessary, https://cmake.org/cmake/help/latest/module/FindCUDA.html
# so we will do it only for older versions of cmake
if(${CMAKE_VERSION} VERSION_LESS ${CMAKE_VERSION_THRESHOLD})
  find_package(CUDA REQUIRED)
endif()  


set(ONNX2TRT_ROOT ${PROJECT_SOURCE_DIR})
# Set C++11 as standard for the whole project
set(CMAKE_CXX_STANDARD  11)

# Enable compiler warnings
if ( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wno-deprecated-declarations -Wno-unused-function")
endif()
if ( MSVC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /W4")
endif()

# Build the libraries with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PARSER_LINKER_SCRIPT  ${ONNX2TRT_ROOT}/libnvonnxparser.version)

#--------------------------------------------------
# Version information
#--------------------------------------------------
set(ONNX2TRT_MAJOR 6)
set(ONNX2TRT_MINOR 0)
set(ONNX2TRT_PATCH 1)

#--------------------------------------------------
# Build configurations, global to all projects
#--------------------------------------------------

set(IMPORTER_SOURCES
  NvOnnxParser.cpp
  ModelImporter.cpp
  builtin_op_importers.cpp
  onnx2trt_utils.cpp
  ShapedWeights.cpp
  OnnxAttrs.cpp
)

set(ONNXIFI_SOURCES onnx_trt_backend.cpp)

set(EXECUTABLE_SOURCES
  main.cpp
)

set(API_TESTS_SOURCES
  getSupportedAPITest.cpp
  ModelImporter.cpp
)


set(HEADERS
  NvOnnxParser.h
)

if (NOT TARGET protobuf::libprotobuf)
  FIND_PACKAGE(Protobuf REQUIRED)
else()
  set(PROTOBUF_LIB "protobuf::libprotobuf")
endif()

if(NOT TARGET onnx_proto)
  # Note: This avoids libprotobuf.so complaining about name collisions at runtime
  if(NOT ONNX_NAMESPACE)
    set(ONNX_NAMESPACE "onnx2trt_onnx")
  endif()
  add_definitions("-DONNX_NAMESPACE=${ONNX_NAMESPACE}")
  add_subdirectory(third_party/onnx EXCLUDE_FROM_ALL)
endif()

# If GPU_ARCHS is user defined, build specifically for specified SM
if (DEFINED GPU_ARCHS)
  message(STATUS "GPU_ARCH defined as ${GPU_ARCHS}. Generating CUDA code for SM ${GPU_ARCHS}")
  separate_arguments(GPU_ARCHS)
# Else list out default SMs to build for.
else()
  message(STATUS "GPU_ARCH is not defined. Generating CUDA code for default SMs.")
  list(APPEND GPU_ARCHS
      35
      53
      61
      70
      )
  # Add SM 75 for CUDA versions >= 10.0
  if (NOT ("${CUDA_VERSION}" VERSION_LESS "10.0"))
    list(APPEND GPU_ARCHS
      75)
  endif()
endif()

set(CUDA_VERBOSE_BUILD ON)

# Generate SASS for each architecture
foreach(arch ${GPU_ARCHS})
    set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()
# Generate PTX for the last architecture
list(GET GPU_ARCHS -1 LATEST_GPU_ARCH)
set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_GPU_ARCH},code=compute_${LATEST_GPU_ARCH}")

if(${CMAKE_VERSION} VERSION_LESS ${CMAKE_VERSION_THRESHOLD})
  if(NOT "${CUDA_NVCC_FLAGS}" MATCHES "-cudart" )
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -cudart static")
  endif()
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} \
    -lineinfo \
    -g \
    --expt-extended-lambda \
    ${GENCODES} \
    ")
  
  if((NOT CXX_VERSION_DEFINED) AND (NOT "${CUDA_NVCC_FLAGS}" MATCHES "-std=c\\+\\+11"))
    list(APPEND CUDA_NVCC_FLAGS -std=c++11)
  endif()

else()
  if(NOT "${CMAKE_CUDA_FLAGS}" MATCHES "-cudart" )
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -cudart static")
  endif()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    -lineinfo \
    -g \
    --expt-extended-lambda \
    ${GENCODES} \
    ")

  if((NOT CXX_VERSION_DEFINED) AND (NOT "${CMAKE_CUDA_FLAGS}" MATCHES "-std=c\\+\\+11"))
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11")
  endif()

endif()  
  


# Specify the cuda host compiler to use the same compiler as cmake.
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})


# CUDNN
set(CUDNN_ROOT_DIR "" CACHE PATH "Folder contains NVIDIA cuDNN")
find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES cuda/include include)
find_library(CUDNN_LIBRARY cudnn
    HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)
find_package_handle_standard_args(
    CUDNN DEFAULT_MSG CUDNN_INCLUDE_DIR CUDNN_LIBRARY)

if(NOT CUDNN_FOUND)
  message(WARNING
      "Cudnn cannot be found. TensorRT depends explicitly "
      "on cudnn so you should consider installing it.")
  return()
endif()

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
find_package_handle_standard_args(
  TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
  message(ERROR "Cannot find TensorRT library.")
endif()


# --------------------------------
# Plugin library
# --------------------------------
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC --expt-extended-lambda")
if(${CMAKE_VERSION} VERSION_LESS ${CMAKE_VERSION_THRESHOLD})
  CUDA_INCLUDE_DIRECTORIES(${CUDNN_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR})
else()
  include_directories(${CUDNN_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR})
endif()

# --------------------------------
# Importer library
# --------------------------------
add_library(nvonnxparser SHARED ${IMPORTER_SOURCES})
target_include_directories(nvonnxparser PUBLIC ${CUDA_INCLUDE_DIRS} ${ONNX_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR})
target_link_libraries(nvonnxparser PUBLIC onnx_proto ${PROTOBUF_LIBRARY} ${CUDNN_LIBRARY} ${TENSORRT_LIBRARY})
set_target_properties(nvonnxparser PROPERTIES
  VERSION   ${ONNX2TRT_MAJOR}.${ONNX2TRT_MINOR}.${ONNX2TRT_PATCH}
  SOVERSION ${ONNX2TRT_MAJOR}
  LINK_DEPENDS ${PARSER_LINKER_SCRIPT}
  LINK_FLAGS "-Wl,--version-script=${PARSER_LINKER_SCRIPT}"
)
add_library(nvonnxparser_static STATIC ${IMPORTER_SOURCES})
target_include_directories(nvonnxparser_static PUBLIC ${CUDA_INCLUDE_DIRS} ${ONNX_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR})
target_link_libraries(nvonnxparser_static PUBLIC onnx_proto ${PROTOBUF_LIBRARY} ${CUDNN_LIBRARY} ${TENSORRT_LIBRARY})

# --------------------------------
# Onnxifi library
# --------------------------------
add_library(trt_onnxify SHARED ${ONNXIFI_SOURCES})
target_include_directories(trt_onnxify PUBLIC ${CUDA_INCLUDE_DIRS} ${ONNX_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR})
target_link_libraries(trt_onnxify PUBLIC nvonnxparser_static ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})

# --------------------------------
# Converter executable
# --------------------------------

add_executable(onnx2trt ${EXECUTABLE_SOURCES})
target_include_directories(onnx2trt PUBLIC ${ONNX_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})

target_link_libraries(onnx2trt PUBLIC ${PROTOBUF_LIB} nvonnxparser_static ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})

# --------------------------------
# API Tests
# --------------------------------

add_executable(getSupportedAPITest ${API_TESTS_SOURCES})
target_include_directories(getSupportedAPITest PUBLIC ${ONNX_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})

target_link_libraries(getSupportedAPITest PUBLIC ${PROTOBUF_LIB} nvonnxparser_static ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})


# --------------------------------
# Installation
# --------------------------------
install(TARGETS onnx2trt
                nvonnxparser
                nvonnxparser_static
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)

install(FILES ${HEADERS}
  DESTINATION include
)

SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Mike Houston") #required
SET(CPACK_PACKAGE_NAME "onnx-trt-dev")
SET(CPACK_PACKAGE_VERSION "0.5.9")
SET(CPACK_PACKAGE_VERSION_MAJOR "0")
SET(CPACK_PACKAGE_VERSION_MINOR "5")
SET(CPACK_PACKAGE_VERSION_PATCH "9")

INCLUDE(CPack)
