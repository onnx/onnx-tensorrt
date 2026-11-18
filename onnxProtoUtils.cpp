/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnxProtoUtils.hpp"

#include <fstream>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <iostream>
#include <onnx/onnx-ml.pb.h>
#include <sstream>

namespace onnx2trt
{
void removeRawDataStrings(std::string& s)
{
    std::string::size_type beg = 0;
    std::string const key = "raw_data: \"";
    std::string const sub = "...";
    while ((beg = s.find(key, beg)) != std::string::npos)
    {
        beg += key.length();
        std::string::size_type end = beg - 1;
        // Skip escaped end-quotes. A quote terminates the string only when preceded by an even number
        // of backslashes: each pair is an escaped backslash, while an odd count escapes the quote
        // itself. Checking just the immediately preceding character would treat `\\"` (an escaped
        // backslash followed by a real quote) as escaped and scan past the terminator.
        while ((end = s.find('"', ++end)) != std::string::npos)
        {
            std::string::size_type backslashes = 0;
            while (end - backslashes > beg && s[end - backslashes - 1] == '\\')
            {
                ++backslashes;
            }
            if (backslashes % 2U == 0U)
            {
                break;
            }
        }
        if (end == std::string::npos)
        {
            // No closing quote: without this check, s[end - 1] above would read out of bounds.
            break;
        }
        if (end - beg > 128)
        { // Only remove large data strings
            s.replace(beg, end - beg, "...");
        }
        beg += sub.length();
    }
}

std::string removeRepeatedDataStrings(std::string const& s)
{
    std::istringstream iss(s);
    std::ostringstream oss;
    bool is_repeat = false;
    for (std::string line; std::getline(iss, line);)
    {
        if (line.find("float_data:") != std::string::npos || line.find("int32_data:") != std::string::npos
            || line.find("int64_data:") != std::string::npos)
        {
            if (!is_repeat)
            {
                is_repeat = true;
                oss << line.substr(0, line.find(':') + 1) << " ...\n";
            }
        }
        else
        {
            is_repeat = false;
            oss << line << "\n";
        }
    }
    return oss.str();
}

std::string onnxIRVersionAsString(int64_t irVersion)
{
    int64_t verMajor = irVersion / 1000000;
    int64_t verMinor = irVersion % 1000000 / 10000;
    int64_t verPatch = irVersion % 10000;
    return (std::to_string(verMajor) + "." + std::to_string(verMinor) + "." + std::to_string(verPatch));
}

} // namespace onnx2trt
