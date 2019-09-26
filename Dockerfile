FROM nvcr.io/nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

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
WORKDIR /usr/local/src
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python2 get-pip.py && \
    pip2 install --upgrade pip && \
    python3 get-pip.py && \
    pip3 install --upgrade pip && \
    rm -f get-pip.py

# Build and install onnx
RUN pip2 install pybind11 && \
    pip2 install protobuf && \
    pip2 install numpy && \
    pip3 install numpy

RUN git clone --recurse-submodules https://github.com/onnx/onnx.git && \
    cd onnx && \
    git checkout dee6d89 && \
    python2 setup.py build && \
    python2 setup.py install && \
    python3 setup.py build && \
    python3 setup.py install && \
    cd ../ && \
    rm -rf onnx/

# Install TensorRT
ENV TENSORRT_VERSION 6.0.1.5
ENV PY3_VERSION 36
COPY TensorRT-${TENSORRT_VERSION}.*.tar.gz .
RUN tar -xvf TensorRT-${TENSORRT_VERSION}.*.tar.gz && \
    cd TensorRT-${TENSORRT_VERSION}/ && \
    cp lib/lib* /usr/lib/x86_64-linux-gnu/ && \
    rm /usr/lib/x86_64-linux-gnu/libnv*.a && \
    cp include/* /usr/include/x86_64-linux-gnu/ && \
    cp bin/* /usr/bin/ && \
    mkdir /usr/share/doc/tensorrt && \
    cp -r doc/* /usr/share/doc/tensorrt/ && \
    mkdir /usr/src/tensorrt && \
    cp -r samples /usr/src/tensorrt/  && \
    pip2 install python/tensorrt-${TENSORRT_VERSION}-cp27-none-linux_x86_64.whl && \
    pip3 install python/tensorrt-${TENSORRT_VERSION}-cp${PY3_VERSION}-none-linux_x86_64.whl && \
    pip2 install uff/uff-*-py2.py3-none-any.whl && \
    pip3 install uff/uff-*-py2.py3-none-any.whl && \
    cd ../ && \
    rm -rf TensorRT-${TENSORRT_VERSION}*

# Build the library

ENV ONNX2TRT_VERSION 0.1.0

WORKDIR /opt/onnx2trt
COPY . .

# For python2.
RUN rm -rf build/ && \
    mkdir build && \
    cd build && \
    cmake -DCUDA_INCLUDE_DIRS=/usr/local/cuda/include/ .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd .. && \
    python2 setup.py build && \
    python2 setup.py install && \
    rm -rf ./build/

# For python3.
RUN mkdir build && \
    cd build && \
    cmake -DCUDA_INCLUDE_DIRS=/usr/local/cuda/include/ .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd .. && \
    python3 setup.py build && \
    python3 setup.py install && \
    rm -rf ./build/

WORKDIR /workspace

RUN cp /opt/onnx2trt/onnx_backend_test.py .

RUN ["/bin/bash"]
