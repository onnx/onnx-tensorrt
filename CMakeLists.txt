# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)
project(onnx2trt LANGUAGES CXX C)

set(ONNX2TRT_ROOT ${PROJECT_SOURCE_DIR})
# Set C++17 as standard for the whole project, as required by ONNX 1.16
set(CMAKE_CXX_STANDARD 17)

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
set(ONNX2TRT_MAJOR 10)
set(ONNX2TRT_MINOR 15)
set(ONNX2TRT_PATCH 1)
set(ONNX2TRT_VERSION "${ONNX2TRT_MAJOR}.${ONNX2TRT_MINOR}.${ONNX2TRT_PATCH}" CACHE STRING "ONNX2TRT version")

#--------------------------------------------------
# Build configurations, global to all projects
#--------------------------------------------------

set(IMPORTER_SOURCES
    AttentionHelpers.cpp
    NvOnnxParser.cpp
    ModelImporter.cpp
    ModelRefitter.cpp
    onnxOpImporters.cpp
    ImporterContext.cpp
    importerUtils.cpp
    ShapedWeights.cpp
    ShapeTensor.cpp
    Status.cpp
    LoopHelpers.cpp
    RNNHelpers.cpp
    OnnxAttrs.cpp
    onnxErrorRecorder.cpp
    ConditionalHelpers.cpp
    bfloat16.cpp
    onnxOpCheckers.cpp
    onnxProtoUtils.cpp
    TensorOrWeights.cpp
    WeightsContext.cpp
    WeightsContextMemoryMap.cpp
    weightUtils.cpp
    errorHelpers.cpp
)

if (BUILD_ONNXIFI)
  set(ONNXIFI_SOURCES onnx_trt_backend.cpp)
endif()

set(API_TESTS_SOURCES
  getSupportedAPITest.cpp
  ModelImporter.cpp
)

# Find protobuf if it's not a target.
if (NOT TARGET protobuf::libprotobuf)
  FIND_PACKAGE(Protobuf)
endif()

# Set protobuf libraries between full / lite.
if (ONNX_USE_LITE_PROTO)
  add_definitions("-DUSE_LITE_PROTOBUF=1")
  set(PROTOBUF_LIBRARY "protobuf::libprotobuf-lite")
else()
  set(PROTOBUF_LIBRARY "protobuf::libprotobuf")
endif()

if(NOT TARGET onnx_proto)
  # Note: This avoids libprotobuf.so complaining about name collisions at runtime
  if(NOT ONNX_NAMESPACE)
    set(ONNX_NAMESPACE "onnx2trt_onnx")
  endif()
  add_definitions("-DONNX_NAMESPACE=${ONNX_NAMESPACE}")
  # Redefine target_compile_features to avoid cross-compilation issues.
  function(target_compile_features)
  endfunction()
  add_subdirectory(third_party/onnx EXCLUDE_FROM_ALL)
endif()

# Find CUDA
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

# TensorRT Python Headers
find_path(TENSORRT_PYTHON_INCLUDE_DIR NvInferPythonPlugin.h
  HINTS ${TENSORRT_ROOT}
  PATH_SUFFIXES include/impl)

# If header is not found, download it from open source release.
if(NOT TENSORRT_PYTHON_INCLUDE_DIR)
  set(PLUGIN_URL "https://raw.githubusercontent.com/NVIDIA/TensorRT/refs/heads/release/${ONNX2TRT_MAJOR}.${ONNX2TRT_MINOR}/include/impl/NvInferPythonPlugin.h")
  set(FILE_DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/NvInferPythonPlugin.h")

  message(NOTICE "Required header NvInferPythonPlugin.h not found. Downloading from ${PLUGIN_URL} to ${FILE_DESTINATION}")

  file(DOWNLOAD ${PLUGIN_URL} ${FILE_DESTINATION}
    SHOW_PROGRESS
    STATUS DOWNLOAD_STATUS
    LOG DOWNLOAD_LOG
  )

  list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
  list(GET DOWNLOAD_STATUS 1 ERROR_MESSAGE)

  if(NOT STATUS_CODE EQUAL 0)
    message(FATAL_ERROR "Error downloading file: ${ERROR_MESSAGE}")
  endif()

  set(TENSORRT_PYTHON_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
endif()

message(NOTICE "Found TensorRT Python headers at ${TENSORRT_PYTHON_INCLUDE_DIR}")

# Output dynamic library names depends on platform:
if (MSVC)
    set(nvonnxparser_lib_name "nvonnxparser_${ONNX2TRT_MAJOR}")
else()
    set(nvonnxparser_lib_name "nvonnxparser")
endif()
# Output static library name is the same cross-platform.
set(nvonnxparser_lib_name_static "nvonnxparser_static")

# --------------------------------
# Importer library
# --------------------------------
add_library(${nvonnxparser_lib_name} SHARED ${IMPORTER_SOURCES})
target_include_directories(${nvonnxparser_lib_name} PUBLIC ${ONNX_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${TENSORRT_PYTHON_INCLUDE_DIR} ${CUDA_INCLUDE_DIR})
target_link_libraries(${nvonnxparser_lib_name} PUBLIC onnx_proto ${PROTOBUF_LIBRARY})
set_target_properties(${nvonnxparser_lib_name} PROPERTIES
  VERSION   ${ONNX2TRT_VERSION}
  SOVERSION ${ONNX2TRT_MAJOR}
  LINK_DEPENDS ${PARSER_LINKER_SCRIPT}
  LINK_FLAGS "-Wl,--version-script=${PARSER_LINKER_SCRIPT}"
  ARCHIVE_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
)
add_library(${nvonnxparser_lib_name_static} STATIC ${IMPORTER_SOURCES})
target_include_directories(${nvonnxparser_lib_name_static} PUBLIC ${ONNX_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${TENSORRT_PYTHON_INCLUDE_DIR} ${CUDA_INCLUDE_DIR})
target_link_libraries(${nvonnxparser_lib_name_static} PUBLIC onnx_proto ${PROTOBUF_LIBRARY})
set_target_properties(${nvonnxparser_lib_name_static} PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
)
# --------------------------------
# Onnxifi library
# --------------------------------
if(BUILD_ONNXIFI)
  add_library(trt_onnxify SHARED ${ONNXIFI_SOURCES})
  target_include_directories(trt_onnxify PUBLIC ${CUDA_INCLUDE_DIR} ${ONNX_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${TENSORRT_PYTHON_INCLUDE_DIR})
  target_link_libraries(trt_onnxify PUBLIC ${nvonnxparser_lib_name_static} ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
endif()

# --------------------------------
# API Tests
# --------------------------------
if (BUILD_API_TEST)
  add_executable(getSupportedAPITest ${API_TESTS_SOURCES})
  target_include_directories(getSupportedAPITest PUBLIC ${ONNX_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
  target_link_libraries(getSupportedAPITest PUBLIC ${PROTOBUF_LIB} ${nvonnxparser_lib_name_static} ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
endif()

# --------------------------------
# Installation
# --------------------------------
install(TARGETS
                ${nvonnxparser_lib_name}
                ${nvonnxparser_lib_name_static}
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
