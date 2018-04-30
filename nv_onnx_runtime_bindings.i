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

/*
 * NvOnnxParserRuntime.i
 * Interface file for generating Python wrapper for
 * the onnx parser class
 */

%module nv_onnx_runtime_bindings
%{
#define SWIG_FILE_WITH_INIT
#include "NvOnnxParserRuntime.h"
%}

%feature("director") nvonnxparser::IPluginFactory;

%rename("%(utitle)s", %$isfunction) "";
%rename("%(regex:/^(I)(.*)/\\2/)s", %$isclass) "";

// CamelCase to snake_case
%rename("%(utitle)s", %$isfunction) "";
// remove I from interface class
%rename("%(regex:/^(I)(.*)/\\2/)s", %$isclass) "";
// support int64_t
%include "stdint.i"

%unrefobject nvonnxparser::IPluginFactory {
  $this->destroy();
}
%newobject nvonnxparser::createPluginFactory;
%delobject nvonnxparser::IPluginFactory::destroy;

%extend nvonnxparser::IPluginFactory {
  // Workaround that allows instances of this class to be passed to functions
  // that expect the base class nvinfer1::IPluginFactory*.
  // TODO: Why does this work in Python as if it's an implicit conversion!?
  nvinfer1::IPluginFactory* _base_ptr_workaround() { return $self; }
}

%include "NvOnnxParserRuntime.h"
