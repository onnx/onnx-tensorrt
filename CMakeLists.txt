# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)
project(onnx2trt LANGUAGES CXX C)

set(ONNX2TRT_ROOT ${PROJECT_SOURCE_DIR})
# Set C++14 as standard for the whole project
set(CMAKE_CXX_STANDARD 14)

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

# Find length of source directory used to pad filename in Status.hpp
string(LENGTH "${CMAKE_SOURCE_DIR}/" SOURCE_LENGTH)
add_definitions("-DSOURCE_LENGTH=${SOURCE_LENGTH}")

#--------------------------------------------------
# Version information
#--------------------------------------------------
set(ONNX2TRT_MAJOR 8)
set(ONNX2TRT_MINOR 5)
set(ONNX2TRT_PATCH 1)
set(ONNX2TRT_VERSION "${ONNX2TRT_MAJOR}.${ONNX2TRT_MINOR}.${ONNX2TRT_PATCH}" CACHE STRING "ONNX2TRT version")

#--------------------------------------------------
# Build configurations, global to all projects
#--------------------------------------------------

set(IMPORTER_SOURCES
  NvOnnxParser.cpp
  ModelImporter.cpp
  builtin_op_importers.cpp
  onnx2trt_utils.cpp
  onnxErrorRecorder.cpp
  ImporterContext.cpp
  ShapedWeights.cpp
  ShapeTensor.cpp
  LoopHelpers.cpp
  RNNHelpers.cpp
  OnnxAttrs.cpp
  ConditionalHelpers.cpp
)

if (BUILD_ONNXIFI)
  set(ONNXIFI_SOURCES onnx_trt_backend.cpp)
endif()

set(API_TESTS_SOURCES
  getSupportedAPITest.cpp
  ModelImporter.cpp
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

# CUDA
if (NOT CUDA_TOOLKIT_ROOT_DIR)
  set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
endif()
find_path(CUDA_INCLUDE_DIR cuda_runtime.h
  HINTS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include
)
MESSAGE(STATUS "Found CUDA headers at ${CUDA_INCLUDE_DIR}")

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
# Importer library
# --------------------------------
add_library(nvonnxparser SHARED ${IMPORTER_SOURCES})
target_include_directories(nvonnxparser PUBLIC ${ONNX_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIR})
target_link_libraries(nvonnxparser PUBLIC onnx_proto ${PROTOBUF_LIBRARY} ${TENSORRT_LIBRARY})
set_target_properties(nvonnxparser PROPERTIES
  VERSION   ${ONNX2TRT_VERSION}
  SOVERSION ${ONNX2TRT_MAJOR}
  LINK_DEPENDS ${PARSER_LINKER_SCRIPT}
  LINK_FLAGS "-Wl,--version-script=${PARSER_LINKER_SCRIPT}"
)
add_library(nvonnxparser_static STATIC ${IMPORTER_SOURCES})
target_include_directories(nvonnxparser_static PUBLIC ${ONNX_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIR})
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
# API Tests
# --------------------------------
if (BUILD_API_TEST)
  add_executable(getSupportedAPITest ${API_TESTS_SOURCES})
  target_include_directories(getSupportedAPITest PUBLIC ${ONNX_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
  target_link_libraries(getSupportedAPITest PUBLIC ${PROTOBUF_LIB} nvonnxparser_static ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
endif()

# --------------------------------
# Installation
# --------------------------------
install(TARGETS
                nvonnxparser
                nvonnxparser_static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)

install(FILES ${HEADERS}
  DESTINATION include
)

SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "NVIDIA") #required
SET(CPACK_PACKAGE_NAME "onnx-trt-dev")
SET(CPACK_PACKAGE_VERSION "0.5.9")
SET(CPACK_PACKAGE_VERSION_MAJOR "0")
SET(CPACK_PACKAGE_VERSION_MINOR "5")
SET(CPACK_PACKAGE_VERSION_PATCH "9")

INCLUDE(CPack)
