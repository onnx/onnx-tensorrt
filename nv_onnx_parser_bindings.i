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
 * NvOnnxParser.i
 * Interface file for generating Python wrapper for
 * the onnx parser class
 */

%module nv_onnx_parser_bindings
%{
#define SWIG_FILE_WITH_INIT
#include "NvOnnxParser.h"
%}

%feature("director") nvonnxparser::IParser;
%feature("director") nvonnxparser::IParserError;

%rename("%(utitle)s", %$isfunction) "";
%rename("%(regex:/^(I)(.*)/\\2/)s", %$isclass) "";

// CamelCase to snake_case
%rename("%(utitle)s", %$isfunction) "";
// remove I from interface class
%rename("%(regex:/^(I)(.*)/\\2/)s", %$isclass) "";
// support int64_t
%include "stdint.i"

%typemap(in) (void const* serialized_onnx_model,
              size_t      serialized_onnx_model_size) {
   if (!PyString_Check($input)) {
       PyErr_SetString(PyExc_ValueError, "Expecting a string");
       return NULL;
   }
   $1 = (void *) PyString_AsString($input);
   $2 = PyString_Size($input);
}

%unrefobject nvonnxparser::IParser {
  $this->destroy();
}
%newobject nvonnxparser::createParser;
%delobject nvonnxparser::IParser::destroy;

%include "NvOnnxParser.h"
