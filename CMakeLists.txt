 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

cmake_minimum_required(VERSION 3.13)
project(onnx2trt LANGUAGES CXX C)

set(ONNX2TRT_ROOT ${PROJECT_SOURCE_DIR})
# Set C++11 as standard for the whole project
set(CMAKE_CXX_STANDARD  11)

# Enable compiler warnings
if (CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wno-deprecated-declarations -Wno-unused-function")
endif()
if (MSVC)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /W4")
endif()

# Build the libraries with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PARSER_LINKER_SCRIPT  ${ONNX2TRT_ROOT}/libnvonnxparser.version)

#--------------------------------------------------
# Version information
#--------------------------------------------------
set(ONNX2TRT_MAJOR 7)
set(ONNX2TRT_MINOR 2)
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
  ShapeTensor.cpp
  LoopHelpers.cpp
  RNNHelpers.cpp
  OnnxAttrs.cpp
)

# Do not build ONNXIFI by default.
if(BUILD_ONNXIFI)
  if (NOT CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
  endif()
  message(debug "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
  find_path(CUDA_INCLUDE_DIR cuda_runtime.h
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include
  )
  set(ONNXIFI_SOURCES onnx_trt_backend.cpp)
endif()

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
if(WIN32)
  find_library(TENSORRT_LIBRARY_MYELIN myelin64_1
    HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
else()
  find_library(TENSORRT_LIBRARY_MYELIN myelin
    HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
endif()
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_MYELIN})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
find_package_handle_standard_args(
  TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
  message(ERROR "Cannot find TensorRT library.")
endif()

# --------------------------------
# Importer library
# --------------------------------
add_library(nvonnxparser SHARED ${IMPORTER_SOURCES})
target_include_directories(nvonnxparser PUBLIC ${ONNX_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})
target_link_libraries(nvonnxparser PUBLIC onnx_proto ${PROTOBUF_LIBRARY} ${TENSORRT_LIBRARY})
set_target_properties(nvonnxparser PROPERTIES
  VERSION   ${ONNX2TRT_MAJOR}.${ONNX2TRT_MINOR}.${ONNX2TRT_PATCH}
  SOVERSION ${ONNX2TRT_MAJOR}
  LINK_DEPENDS ${PARSER_LINKER_SCRIPT}
  LINK_FLAGS "-Wl,--version-script=${PARSER_LINKER_SCRIPT}"
)
add_library(nvonnxparser_static STATIC ${IMPORTER_SOURCES})
target_include_directories(nvonnxparser_static PUBLIC ${ONNX_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})
target_link_libraries(nvonnxparser_static PUBLIC onnx_proto ${PROTOBUF_LIBRARY} ${TENSORRT_LIBRARY})

# --------------------------------
# Onnxifi library
# --------------------------------
if(BUILD_ONNXIFI)
  add_library(trt_onnxify SHARED ${ONNXIFI_SOURCES})
  target_include_directories(trt_onnxify PUBLIC ${CUDA_INCLUDE_DIR} ${ONNX_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(trt_onnxify PUBLIC nvonnxparser_static ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
endif()

# --------------------------------
# Converter executable
# --------------------------------
add_executable(onnx2trt ${EXECUTABLE_SOURCES})
target_include_directories(onnx2trt PUBLIC ${ONNX_INCLUDE_DIRS})
target_link_libraries(onnx2trt PUBLIC ${PROTOBUF_LIB} onnx nvonnxparser_static ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS}) #${CUDA_LIBRARIES} 

# --------------------------------
# API Tests
# --------------------------------
add_executable(getSupportedAPITest ${API_TESTS_SOURCES})
target_include_directories(getSupportedAPITest PUBLIC ${ONNX_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
target_link_libraries(getSupportedAPITest PUBLIC ${PROTOBUF_LIB} nvonnxparser_static ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS}) #${CUDA_LIBRARIES} 

# --------------------------------
# Installation
# --------------------------------
install(TARGETS
                onnx2trt
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
