FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN rm -f /usr/lib/x86_64-linux-gnu/libnccl_static.a \
          /usr/lib/x86_64-linux-gnu/libcudnn_static_v7.a

# Install package dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        wget \
        git \
        python \
        python3 \
        python-dev \
        python3-dev \
        libprotobuf-dev \
        protobuf-compiler \
        cmake \
        swig \
    && rm -rf /var/lib/apt/lists/*

# Install pip
RUN cd /usr/local/src && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python2 get-pip.py && \
    pip2 install --upgrade pip && \
    python3 get-pip.py && \
    pip3 install --upgrade pip && \
    rm -f get-pip.py

# Build and install onnx
RUN cd /usr/local/src && \
    git clone --recurse-submodules https://github.com/onnx/onnx.git && \
    cd onnx && \
    git checkout dee6d89 && \
    pip2 install pybind11 && \
    pip2 install protobuf && \
    pip2 install numpy && \
    pip3 install numpy && \
    python setup.py build && \
    python setup.py install && \
    cd ../ && \
    rm -rf onnx/

# Install TensorRT
WORKDIR /usr/local/src
ENV TENSORRT_VERSION 3.0
COPY TensorRT-${TENSORRT_VERSION}.*.tar.gz .
RUN tar -xvf TensorRT-${TENSORRT_VERSION}.*.tar.gz && \
    cd TensorRT-${TENSORRT_VERSION}.* && \
    cp lib/* /usr/lib/x86_64-linux-gnu/ && \
    rm /usr/lib/x86_64-linux-gnu/libnv*.a && \
    cp include/* /usr/include/x86_64-linux-gnu/ && \
    cp bin/* /usr/bin/ && \
    mkdir /usr/share/doc/tensorrt && \
    cp -r doc/* /usr/share/doc/tensorrt/ && \
    mkdir /usr/src/tensorrt && \
    cp -r samples /usr/src/tensorrt/  && \
    pip2 install python/tensorrt-${TENSORRT_VERSION}.*-cp27-cp27mu-linux_x86_64.whl && \
    pip3 install python/tensorrt-${TENSORRT_VERSION}.*-cp35-cp35m-linux_x86_64.whl && \
    pip2 install uff/uff-*-py2.py3-none-any.whl && \
    pip3 install uff/uff-*-py2.py3-none-any.whl && \
    cd ../ && \
    rm -rf TensorRT-${TENSORRT_VERSION}.*

# Build the library

ENV ONNX2TRT_VERSION 0.1.0

WORKDIR /opt/onnx2trt
COPY . .
RUN rm -rf build/ && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd .. && \
    python setup.py build && \
    python setup.py install && \
    rm -rf ./build/

WORKDIR /workspace

RUN cp /opt/onnx2trt/onnx_backend_test.py .

RUN ["/bin/bash"]
