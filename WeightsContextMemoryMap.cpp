/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "WeightsContext.hpp"
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace onnx2trt
{
int64_t getFileSize(std::string const& file)
{
    std::ifstream fileStream(file, std::ios::binary);
    if (!fileStream)
    {
        return -1L;
    }
    fileStream.seekg(0, std::ios::end);
    std::streamsize fileSize = fileStream.tellg();
    return static_cast<int64_t>(fileSize);
}

#ifdef _WIN32
WeightsContext::MemoryMapping_t WeightsContext::mmap(std::string const& file)
{
    auto* ctx = this; // For logging macros.

    auto it = mMemoryMappings.find(file);
    if (it != mMemoryMappings.end())
    {
        return it->second;
    }

    int64_t fileSize = getFileSize(file);

    if (fileSize < 0L)
    {
        LOG_ERROR("Failed to open file: " << file);
        return {nullptr, -1L};
    }

    FileHandle fd
        = CreateFileA(file.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

    if (fd == INVALID_HANDLE_VALUE)
    {
        LOG_ERROR("Failed to open file: " << file);
        return {nullptr, -1L};
    }

    FileHandle mappingHandle = CreateFileMapping(fd, nullptr, PAGE_READONLY, 0U, 0U, nullptr);

    if (mappingHandle == INVALID_HANDLE_VALUE)
    {
        LOG_ERROR("Failed to map file to memory: " << file);
        CloseHandle(fd);
        return {nullptr, -1L};
    }

    auto const mappedAddr = MapViewOfFile(mappingHandle, FILE_MAP_READ, 0U, 0U, 0U);

    if (mappedAddr == nullptr)
    {
        LOG_ERROR("Failed to map file to memory: " << file);
        CloseHandle(fd);
        CloseHandle(mappingHandle);
        return {nullptr, -1L};
    }

    mMappedFiles[file] = fd;
    mFileMappingHandles[file] = mappingHandle;
    mMemoryMappings[file] = std::make_pair(mappedAddr, fileSize);
    return std::make_pair(mappedAddr, fileSize);
}

void WeightsContext::clearMemoryMappings()
{
    for (auto const& [file, mapping] : mMemoryMappings)
    {
        UnmapViewOfFile(mapping.first);
    }

    for (auto const& [file, fd] : mFileMappingHandles)
    {
        CloseHandle(fd);
    }

    for (auto const& [file, fd] : mMappedFiles)
    {
        CloseHandle(fd);
    }

    mMappedFiles.clear();
    mFileMappingHandles.clear();
    mMemoryMappings.clear();
}
#else
WeightsContext::MemoryMapping_t WeightsContext::mmap(std::string const& file)
{
    auto* ctx = this; // For logging macros.

    auto it = mMemoryMappings.find(file);
    if (it != mMemoryMappings.end())
    {
        return it->second;
    }

    int64_t fileSize = getFileSize(file);

    if (fileSize < 0L)
    {
        LOG_ERROR("Failed to open file: " << file);
        return {nullptr, -1L};
    }

    FileHandle fd = open(file.c_str(), O_RDONLY);

    if (fd == -1L)
    {
        LOG_ERROR("Failed to open file: " << file);
        return {nullptr, -1L};
    }

    void* mappedAddr = ::mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);

    if (mappedAddr == MAP_FAILED)
    {
        LOG_ERROR("Failed to map file to memory: " << file);
        close(fd);
        return {nullptr, -1L};
    }

    mMappedFiles[file] = fd;
    mMemoryMappings[file] = std::make_pair(mappedAddr, fileSize);
    return std::make_pair(mappedAddr, fileSize);
}

void WeightsContext::clearMemoryMappings()
{
    for (auto const& [file, mapping] : mMemoryMappings)
    {
        ::munmap(mapping.first, mapping.second);
    }

    for (auto const& [file, fd] : mMappedFiles)
    {
        close(fd);
    }

    mMappedFiles.clear();
    mMemoryMappings.clear();
}
#endif
} // namespace onnx2trt
