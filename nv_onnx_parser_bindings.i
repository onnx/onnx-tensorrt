/*
 * SPDX-License-Identifier: Apache-2.0
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
