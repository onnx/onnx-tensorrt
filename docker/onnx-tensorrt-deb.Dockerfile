FROM nvcr.io/nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
ARG TENSORRT_VERSION=6.0.1.5

# Install package dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        wget \
        git \
        python \
        python-dev \
        python-pip \
        python-setuptools \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        libprotobuf-dev \
        protobuf-compiler \
        cmake \
        swig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/onnx-tensorrt
COPY . .

RUN dpkg -i nv-tensorrt-repo-ubuntu1804-cuda10.1-trt${TENSORRT_VERSION}-ga-20190913_1-1_amd64.deb&& \
    apt-key add /var/nv-tensorrt-repo-cuda10.1-trt${TENSORRT_VERSION}-ga-20190913/7fa2af80.pub && \
    apt-get update && \
    apt-get install -y tensorrt && \
    apt-get install -y python-libnvinfer-dev && \
    apt-get install -y python3-libnvinfer-dev && \
    apt-get install -y uff-converter-tf && \
    rm nv-tensorrt-repo-ubuntu1804-cuda10.1-trt${TENSORRT_VERSION}-ga-20190913_1-1_amd64.deb
RUN dpkg -l | grep TensorRT

# Build and install onnx
RUN pip2 install onnx==1.5 pytest==4.6.5
RUN pip3 install onnx==1.5 pytest==5.1.2

# Build the library
ENV ONNX2TRT_VERSION 0.1.0

WORKDIR /opt/onnx-tensorrt

RUN rm -rf build/ && \
    mkdir -p build && \
    cd build && \
    cmake -DCUDA_INCLUDE_DIRS=/usr/local/cuda/include/ .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd .. && \
    # For python2.
    python2 setup.py build && \
    python2 setup.py install && \
    # For python3.
    python3 setup.py build && \
    python3 setup.py install && \
    rm -rf ./build/

WORKDIR /workspace

RUN cp /opt/onnx-tensorrt/onnx_backend_test.py .

RUN ["/bin/bash"]
