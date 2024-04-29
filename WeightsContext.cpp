/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "WeightsContext.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>

namespace onnx2trt
{

void* WeightsContext::ownWeights(
    void const* weightValues, const ShapedWeights::DataType dataType, nvinfer1::Dims const& shape, const size_t nBytes)
{
    void* reservedWeights{createTempWeights(dataType, shape).values};
    std::memcpy(reservedWeights, weightValues, nBytes);
    return reservedWeights;
}

int32_t* WeightsContext::convertUINT8(uint8_t const* weightValues, nvinfer1::Dims const& shape)
{
    int64_t const nbWeights = volume(shape);
    int32_t* int32Weights{
        reinterpret_cast<int32_t*>(createTempWeights(::ONNX_NAMESPACE::TensorProto::INT32, shape).values)};

    for (int64_t i = 0; i < nbWeights; i++)
    {
        int32Weights[i] = static_cast<int32_t>(weightValues[i]);
    }
    return int32Weights;
}

float* WeightsContext::convertDouble(double const* weightValues, nvinfer1::Dims const& shape)
{
    auto* ctx = this; // For logging macros.
    int64_t const nbWeights = volume(shape);
    float* floatWeights{
        reinterpret_cast<float*>(createTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, shape).values)};

    bool outOfBounds{false};
    double const floatMax = static_cast<double>(std::numeric_limits<float>::max());
    double const floatMin = static_cast<double>(std::numeric_limits<float>::lowest());
    for (int64_t i = 0; i < nbWeights; i++)
    {
        if (weightValues[i] > floatMax || weightValues[i] < floatMin)
        {
            floatWeights[i] = static_cast<float>(std::max(std::min(weightValues[i], floatMax), floatMin));
            LOG_WARNING("Weight at index " << i << ": " << weightValues[i]
                                        << " is out of range. Clamping to: " << floatWeights[i]);
            outOfBounds = true;
        }
        else
        {
            floatWeights[i] = static_cast<float>(weightValues[i]);
        }
    }
    if (outOfBounds)
    {
        LOG_WARNING("One or more weights outside the range of FLOAT was clamped");
    }

    return floatWeights;
}

uint8_t* WeightsContext::convertPackedInt32Data(
    int32_t const* weightValues, nvinfer1::Dims const& shape, size_t nbytes, int32_t onnxdtype)
{
    uint8_t* newWeights{static_cast<uint8_t*>(createTempWeights(onnxdtype, shape).values)};

    for (size_t i = 0; i < nbytes; i++)
    {
        newWeights[i] = static_cast<uint8_t>(weightValues[i]);
    }
    return newWeights;
}

// Helper function to validate size_t multiplications will not overflow
bool multiplicationWillOverflow(size_t const a, size_t const b)
{
    if (b == 0)
    {
        return false;
    }
    if (a > std::numeric_limits<size_t>::max() / b)
    {
        return true;
    }
    return false;
}

// Helper function to ensure that a ONNX initializer is supportable by TensorRT.
bool validateOnnxInitializer(::ONNX_NAMESPACE::TensorProto const& onnxTensor)
{
    // Validate type.
    auto onnxDtype = onnxTensor.data_type();
    auto typeSize = getDtypeSizeBits(onnxDtype);
    if (typeSize == -1 || typeSize == 0)
    {
        return false;
    }
    // Validate rank.
    auto nbDims = onnxTensor.dims().size();
    if (nbDims > nvinfer1::Dims::MAX_DIMS)
    {
        return false;
    }
    // Validate volume is within bounds.
    size_t vol = 1;
    for (int32_t i = 0; i < nbDims; i++)
    {
        auto dimVal = onnxTensor.dims().Get(i);
        if (dimVal == 0)
        {
            vol = 0;
            break;
        }
        if (vol > std::numeric_limits<size_t>::max() / dimVal)
        {
            return false;
        }
        vol = vol * dimVal;
    }
    // Validate size in bytes is within bounds.
    if (vol > std::numeric_limits<size_t>::max() / typeSize)
    {
        return false;
    }

    return true;
}

// Function to read bytes from an external file and return the data in a buffer.
bool WeightsContext::parseExternalWeights(
    std::string const& file, int64_t offset, int64_t length, std::vector<char>& weightsBuf, size_t& size)
{
    auto* ctx = this; // For logging macros.
    // Accessing parent directories (i.e. ../) is not allowed. Normalize path first.
    auto path = mOnnxFileLocation;
    std::string normalizedFile = normalizePath(file);
    bool illegalDir{false};
#ifdef _MSC_VER
    illegalDir |= normalizedFile.find("..\\") != std::string::npos;
#endif
    illegalDir |= normalizedFile.find("../") != std::string::npos;

    if (illegalDir)
    {
        LOG_ERROR("Relative paths to parent (../) are not allowed in ONNX external weights! Normalized path is: "
            << normalizedFile);
        return false;
    }
    // The weight paths in the ONNX model are relative paths to the main ONNX file.
#ifdef _MSC_VER
    size_t slash = path.rfind("\\");
    // When using WSL path can have "\" or "/". Need to check both options here.
    if (slash == std::string::npos)
    {
        slash = path.rfind("/");
    }
#else
    size_t slash = path.rfind("/");
#endif
    if (slash != std::string::npos)
    {
        path.replace(slash + 1, path.size() - (slash + 1), normalizedFile);
    }
    else
    {
        path = normalizedFile;
    }
    LOG_VERBOSE("Reading weights from external file: " << path);
    std::ifstream relPathFile(path, std::ios::binary | std::ios::ate);
    if (!relPathFile)
    {
        LOG_ERROR("Failed to open file: " << path);
        return false;
    }
    std::streamsize fileSize = relPathFile.tellg();
    relPathFile.seekg(offset, std::ios::beg);
    int64_t weightsBufSize = length == 0 ? fileSize : length;
    weightsBuf.resize(weightsBufSize);
    if (!relPathFile.read(weightsBuf.data(), weightsBuf.size()))
    {
        LOG_ERROR("Failed to read weights from external file: " << path);
        return false;
    }
    size = weightsBuf.size();
    return true;
}

// Function to read data from an ONNX Tensor and move it into a ShapedWeights object. Handles external weights as well.
bool WeightsContext::convertOnnxWeights(
    ::ONNX_NAMESPACE::TensorProto const& onnxTensor, ShapedWeights* weights, bool ownAllWeights)
{
    auto* ctx = this; // For logging macros.

    // Sanity check for onnxTensors
    if (!validateOnnxInitializer(onnxTensor))
    {
        LOG_ERROR("ONNX initializer " << onnxTensor.name() << " cannot be imported into TensorRT!");
        return false;
    }

    void* dataPtr{nullptr};
    size_t nbytes{0};
    auto onnxDtype = onnxTensor.data_type();

    nvinfer1::Dims shape{};
    shape.nbDims = onnxTensor.dims().size();
    std::copy_n(onnxTensor.dims().begin(), shape.nbDims, shape.d);
    // ONNX weight values can be stored in either the TensorProto itself, or in an external file in the case
    // of large models. Check for this here.
    auto dataLocation = onnxTensor.data_location();
    // External Data
    if (dataLocation == 1)
    {
        std::string location{""};
        int64_t offset{0};
        int64_t length{0};

        // onnxTensor.external_data() is a String : String map that holds metadata about how to read from an external
        // file
        for (auto onnxMapEntry : onnxTensor.external_data())
        {
            auto keyName = onnxMapEntry.key();
            if (keyName == "location")
            {
                location = onnxMapEntry.value();
            }
            else if (keyName == "offset")
            {
                offset = std::atoll(onnxMapEntry.value().c_str());
            }
            else if (keyName == "length")
            {
                length = std::atoll(onnxMapEntry.value().c_str());
            }
            // Not used at the moment
            else if (keyName == "checksum")
            {
                continue;
            }
            else
            {
                LOG_ERROR("Key value of: " << keyName << " was not expected!");
                return false;
            }
        }

        // Buffer to hold the data read from the file
        std::vector<char> dataBuf;
        // Will update dataBuf and nbytes by reference.
        if (!parseExternalWeights(location, offset, length, dataBuf, nbytes))
        {
            return false;
        }

        // For weights parsed from external files, createTempWeights is necessary to keep them in scope
        ShapedWeights externalWeights;
        dataPtr = dataBuf.data();

        // Cast non-native TRT types to their corresponding proxy types
        if (onnxDtype == ::ONNX_NAMESPACE::TensorProto::UINT8)
        {
            // Cast UINT8 weights to INT32.
            dataPtr = convertUINT8(reinterpret_cast<uint8_t const*>(dataPtr), shape);
            size_t const sizeOffset = sizeof(int32_t) / sizeof(uint8_t);
            if (multiplicationWillOverflow(nbytes, sizeOffset))
            {
                return false;
            }
            nbytes = nbytes * sizeOffset;
            onnxDtype = ::ONNX_NAMESPACE::TensorProto::INT32;
        }
        else if (onnxDtype == ::ONNX_NAMESPACE::TensorProto::DOUBLE)
        {
            // Cast DOUBLE weights to FLOAT.
            dataPtr = convertDouble(reinterpret_cast<double const*>(dataPtr), shape);
            nbytes = nbytes / (sizeof(double) / sizeof(float));
            onnxDtype = ::ONNX_NAMESPACE::TensorProto::FLOAT;
        }

        // Create the holder for external weights.
        externalWeights = createTempWeights(onnxDtype, shape);

        // Check if the size of external weights is as expected.
        if (externalWeights.size_bytes() != nbytes)
        {
            LOG_ERROR("Unexpected size for the external weights! Expected size: "
                << externalWeights.size_bytes() << " bytes (shape = " << shape << "). Actual size: " << nbytes
                << " bytes.");
            return false;
        }

        // Copy the weight values into externalWeights.
        std::memcpy(externalWeights.values, dataPtr, nbytes);

        *weights = externalWeights;
        return true;
    }

    // Weights information is within the TensorProto itself

    // Cast non-native TRT types to their corresponding proxy types
    if (onnxDtype == ::ONNX_NAMESPACE::TensorProto::UINT8)
    {
        onnxDtype = ::ONNX_NAMESPACE::TensorProto::INT32;
        if (onnxTensor.raw_data().size() > 0)
        {
            dataPtr = convertUINT8(reinterpret_cast<uint8_t const*>(onnxTensor.raw_data().data()), shape);
            size_t const sizeOffset = (sizeof(int32_t) / sizeof(uint8_t));
            if (multiplicationWillOverflow(nbytes, sizeOffset))
            {
                return false;
            }
            nbytes = onnxTensor.raw_data().size() * sizeOffset;
        }
        else if (onnxTensor.int32_data().size() > 0)
        {
            dataPtr = (void*) onnxTensor.int32_data().data();
            if (multiplicationWillOverflow(nbytes, sizeof(int32_t)))
            {
                return false;
            }
            nbytes = onnxTensor.int32_data().size() * sizeof(int32_t);
            if (ownAllWeights)
            {
                dataPtr = ownWeights(dataPtr, onnxDtype, shape, nbytes);
            }
        }
    }
    else if (onnxDtype == ::ONNX_NAMESPACE::TensorProto::DOUBLE)
    {
        if (onnxTensor.raw_data().size() > 0)
        {
            dataPtr = convertDouble(reinterpret_cast<double const*>(onnxTensor.raw_data().data()), shape);
            nbytes = onnxTensor.raw_data().size() / (sizeof(double) / sizeof(float));
        }
        else if (onnxTensor.double_data().size() > 0)
        {
            dataPtr = convertDouble(onnxTensor.double_data().data(), shape);
            if (multiplicationWillOverflow(nbytes, sizeof(float)))
            {
                return false;
            }
            nbytes = onnxTensor.double_data().size() * sizeof(float);
        }
        onnxDtype = ::ONNX_NAMESPACE::TensorProto::FLOAT;
    }

    // Check for supported types that can be found in the int32_data field in the TensorProto
    // https://github.com/onnx/onnx/blob/609282efe8d4871f620141223139bbb99bdbe9f6/onnx/onnx.proto#L567
    else if (onnxDtype == ::ONNX_NAMESPACE::TensorProto::INT32 || onnxDtype == ::ONNX_NAMESPACE::TensorProto::INT64
        || onnxDtype == ::ONNX_NAMESPACE::TensorProto::FLOAT16 || onnxDtype == ::ONNX_NAMESPACE::TensorProto::BFLOAT16
        || onnxDtype == ::ONNX_NAMESPACE::TensorProto::INT8 || onnxDtype == ::ONNX_NAMESPACE::TensorProto::BOOL
        || onnxDtype == ::ONNX_NAMESPACE::TensorProto::INT4)
    {
        if (onnxTensor.raw_data().size() > 0)
        {
            dataPtr = (void*) (onnxTensor.raw_data().data());
            nbytes = onnxTensor.raw_data().size();
            if (ownAllWeights)
            {
                dataPtr = ownWeights(dataPtr, onnxDtype, shape, nbytes);
            }
        }
        else
        {
            nbytes = getTensorOrWeightsSizeBytes(onnxTensor.int32_data().size(), onnxDtype);
            switch (onnxDtype)
            {
            case ::ONNX_NAMESPACE::TensorProto::INT32:
                dataPtr = (void*) (onnxTensor.int32_data().data());
                if (ownAllWeights)
                {
                    dataPtr = ownWeights(dataPtr, onnxDtype, shape, nbytes);
                }
                break;
            case ::ONNX_NAMESPACE::TensorProto::INT64:
                nbytes = getTensorOrWeightsSizeBytes(onnxTensor.int64_data().size(), onnxDtype);
                dataPtr = (void*) (onnxTensor.int64_data().data());
                if (ownAllWeights)
                {
                    dataPtr = ownWeights(dataPtr, onnxDtype, shape, nbytes);
                }
                break;
            case ::ONNX_NAMESPACE::TensorProto::FLOAT16:
            case ::ONNX_NAMESPACE::TensorProto::BFLOAT16:
                dataPtr = convertInt32Data<uint16_t>(onnxTensor.int32_data().data(), shape, onnxDtype);
                break;
            case ::ONNX_NAMESPACE::TensorProto::INT8:
                dataPtr = convertInt32Data<int8_t>(onnxTensor.int32_data().data(), shape, onnxDtype);
                break;
            case ::ONNX_NAMESPACE::TensorProto::BOOL:
                dataPtr = convertInt32Data<uint8_t>(onnxTensor.int32_data().data(), shape, onnxDtype);
                break;
            case ::ONNX_NAMESPACE::TensorProto::INT4:
                // int4 data is packed, each int32 element contains one byte (two int4 nibbles)
                nbytes = onnxTensor.int32_data().size();
                dataPtr = convertPackedInt32Data(onnxTensor.int32_data().data(), shape, nbytes, onnxDtype);
                break;
            default:
                LOG_ERROR("Found unsupported datatype (" << onnxDtype
                                                         << ") when importing initializer: " << onnxTensor.name());
                break;
            }
        }
    }
    else if (onnxDtype == ::ONNX_NAMESPACE::TensorProto::FLOAT)
    {
        if (onnxTensor.raw_data().size() > 0)
        {
            dataPtr = (void*) (onnxTensor.raw_data().data());
            nbytes = onnxTensor.raw_data().size();
        }
        else
        {
            dataPtr = (void*) (onnxTensor.float_data().data());
            if (multiplicationWillOverflow(nbytes, sizeof(float)))
            {
                return false;
            }
            nbytes = onnxTensor.float_data().size() * sizeof(float);
        }
        if (ownAllWeights)
        {
            dataPtr = ownWeights(dataPtr, onnxDtype, shape, nbytes);
        }
    }
    else if (onnxDtype == ::ONNX_NAMESPACE::TensorProto::FLOAT8E4M3FN)
    {
        assert(onnxTensor.int32_data().size());
        dataPtr = (void*) (onnxTensor.int32_data().data());
        nbytes = onnxTensor.int32_data().size();
        if (ownAllWeights)
        {
            dataPtr = ownWeights(dataPtr, onnxDtype, shape, nbytes);
        }
    }
    else
    {
        LOG_ERROR("Found unsupported datatype (" << onnxDtype << ") when importing initializer: " << onnxTensor.name());
        return false;
    }
    onnx2trt::ShapedWeights trt_weights(onnxDtype, dataPtr, shape);
    // Sanity check that weights were converted properly
    if (trt_weights.size_bytes() != nbytes)
    {
        LOG_ERROR("Size mismatch when importing initializer: " << onnxTensor.name() << ". Expected size: " << nbytes
                                                            << " , actual size: " << trt_weights.size_bytes());
        return false;
    }
    *weights = trt_weights;
    return true;
}

float* WeightsContext::convertFP16Data(void* weightValues, nvinfer1::Dims const& shape)
{
    int64_t const nbWeights = volume(shape);
    float* newWeights{static_cast<float*>(createTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, shape).values)};

    half_float::half* tempValues = static_cast<half_float::half*>(weightValues);

    for (int64_t i = 0; i < nbWeights; i++)
    {
        newWeights[i] = tempValues[i];
    }
    return newWeights;
}

float* WeightsContext::getFP32Values(ShapedWeights const& w)
{
    assert((w.type == ::ONNX_NAMESPACE::TensorProto::FLOAT || w.type == ::ONNX_NAMESPACE::TensorProto::FLOAT16)
        && "Conversion only valid from FLOAT or FLOAT16");
    return (w.type == ::ONNX_NAMESPACE::TensorProto::FLOAT) ? static_cast<float*>(w.values)
                                                            : convertFP16Data(w.values, w.shape);
}

ShapedWeights WeightsContext::createNamedTempWeights(ShapedWeights::DataType type, nvinfer1::Dims const& shape,
    std::set<std::string>& namesSet, int64_t& suffixCounter, bool batchNormNode)
{
    std::string const& name
        = generateUniqueName(namesSet, suffixCounter, batchNormNode ? "tmp_batch_norm_weight" : "tmp_weight");
    return createNamedWeights(type, shape, name);
}

ShapedWeights WeightsContext::createNamedWeights(
    ShapedWeights::DataType type, nvinfer1::Dims const& shape, std::string const& name)
{
    ShapedWeights weights(type, nullptr, shape);
    weights.setName(name.c_str());
    allocateTempWeights(weights);
    return weights;
}

} // namespace onnx2trt
