FROM nvcr.io/nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04


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
        python-dev \
        python-pip \
        python3 \
        python3-dev \
        python3-pip \
        libprotobuf-dev \
        protobuf-compiler \
        cmake \
        swig \
    && rm -rf /var/lib/apt/lists/*


# Build and install onnx
RUN pip2 install onnx==1.5 
RUN pip2 install pytest==4.6.5
RUN pip3 install onnx==1.5 pytest==5.1.2

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
    cmake .. && \
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
    cmake .. && \
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
