/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvInfer.h>
#include <memory>

#if NV_TENSORRT_MAJOR < 4
namespace nvinfer1
{

enum class PluginFormat : uint8_t
{
    kNCHW = 0,   //!< NCHW
    kNC2HW2 = 1, //!< NCHW with 2-element packed channels
    kNHWC8 = 2   //!< NHWC with 8-element packed channels (C must be a multiple of 8)
};
// from NvInfer.h
class IPluginExt : public IPlugin
{
public:
    virtual int getTensorRTVersion() const noexcept
    {
        return NV_TENSORRT_VERSION;
    }
    virtual bool supportsFormat(DataType type, PluginFormat format) const noexcept = 0;
    virtual void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        DataType type, PluginFormat format, int maxBatchSize) noexcept
        = 0;

protected:
    void configure(
        const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) noexcept final
    {
        try
        {
            DataType type = nvinfer1::DataType::kFLOAT;
            PluginFormat format = nvinfer1::PluginFormat::kLINEAR;
            return this->configureWithFormat(inputDims, nbInputs, outputDims, nbOutputs, type, format, maxBatchSize);
        }
        catch (const std::exception& e)
        {
            nvinfer1::getLogger()->log(nvinfer1::ILogger::Severity::kERROR, e.what().c_str());
        }
    }
    virtual ~IPluginExt()
    {
    }
};

} // namespace nvinfer1
#endif

namespace onnx2trt
{

struct IOwnable
{
    virtual void destroy() = 0;

protected:
    virtual ~IOwnable()
    {
    }
};

struct OwnableDeleter
{
    void operator()(IOwnable* obj) const
    {
        obj->destroy();
    }
};

using UniqueOwnable = std::unique_ptr<IOwnable, OwnableDeleter>;
class Plugin;
class PluginV2;

} // namespace onnx2trt
