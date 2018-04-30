/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef NV_ONNX_PARSER_RUNTIME_H
#define NV_ONNX_PARSER_RUNTIME_H

#include "NvOnnxParser.h"

namespace nvonnxparser
{

 /** \class IPluginFactory
 *
 * \brief a destroyable plugin factory object
 */
class IPluginFactory : public nvinfer1::IPluginFactory
{
public:
    /** \brief destroy this object
     */
    virtual void destroy() = 0;
protected:
    virtual ~IPluginFactory() {}
};

} // namespace nvonnxparser

extern "C" void* createNvOnnxParserPluginFactory_INTERNAL(void* logger, int version);

namespace nvonnxparser
{

#ifdef SWIG
inline IPluginFactory* createPluginFactory(nvinfer1::ILogger* logger)
{
    return static_cast<IPluginFactory*>(
        createNvOnnxParserPluginFactory_INTERNAL(logger, NV_ONNX_PARSER_VERSION));
}
#endif // SWIG

namespace
{

/** \brief Create a new plugin factory for deserializing engines built using
 *         the ONNX parser.
 *
 * This plugin factory handles deserialization of the plugins that are built
 * into the ONNX parser. Engines built using the ONNX parser must use this
 * plugin factory during deserialization.
 *
 * \param logger The logger to use
 *
 * \return a new plugin factory object or NULL if an error occurred
 * \see IPluginFactory
 */
inline IPluginFactory* createPluginFactory(nvinfer1::ILogger& logger)
{
    return static_cast<IPluginFactory*>(
        createNvOnnxParserPluginFactory_INTERNAL(&logger, NV_ONNX_PARSER_VERSION));
}

} // namespace

} // namespace nvonnxparser

#endif // NV_ONNX_PARSER_RUNTIME_H
