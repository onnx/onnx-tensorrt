#include "Status.hpp"

namespace onnx2trt
{

template <typename T>
std::ostream& printSequence(std::ostream& stream, const T* begin, int count)
{
    stream << "(";
    if (count > 0)
    {
        std::copy_n(begin, count - 1, std::ostream_iterator<T>(stream, ", "));
        stream << begin[count - 1];
    }
    stream << ")";
    return stream;
}

std::ostream& operator<<(std::ostream& stream, nvinfer1::Dims const& shape)
{
    return printSequence(stream, shape.d, shape.nbDims);
}

std::ostream& operator<<(std::ostream& stream, nvinfer1::Permutation const& perm)
{
    return printSequence(stream, perm.order, nvinfer1::Dims::MAX_DIMS);
}

std::ostream& operator<<(std::ostream& stream, nvinfer1::DataType const& dtype)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT: return stream << "float32";
    case nvinfer1::DataType::kHALF: return stream << "float16";
    case nvinfer1::DataType::kBF16: return stream << "bfloat16";
    case nvinfer1::DataType::kINT8: return stream << "int8";
    case nvinfer1::DataType::kUINT8: return stream << "uint8";
    case nvinfer1::DataType::kINT32: return stream << "int32";
    case nvinfer1::DataType::kINT64: return stream << "int64";
    case nvinfer1::DataType::kBOOL: return stream << "bool";
    case nvinfer1::DataType::kFP8: return stream << "float8";
    case nvinfer1::DataType::kE8M0: return stream << "floatE8M0";
    case nvinfer1::DataType::kINT4: return stream << "int4";
    case nvinfer1::DataType::kFP4: return stream << "fp4";

    default: throw std::runtime_error("Unknown dtype");
    }
}

} // namespace onnx2trt
