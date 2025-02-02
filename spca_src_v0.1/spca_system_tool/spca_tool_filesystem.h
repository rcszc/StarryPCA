// spca_tool_filesystem. [2023_12_03] RCSZ.
// 2024_03_26: ÒÆÖ² ext_fileloader => spca_tool_filesystem RCSZ.
// update: 2025_03_02.

#ifndef _SPCA_TOOL_FILESYSTEM_H
#define _SPCA_TOOL_FILESYSTEM_H
#include <string>
#include <vector>
#include <fstream>

class FileLoaderBinary {
protected:
    std::vector<uint8_t> ReadBinaryDataTemp = {};
    size_t               ReadBinaryDataSize = {};
public:
    // true:success, false:failed.
    bool ReadBinaryFile(const std::string& filename);

    // non-chache.
    bool WriterBinaryFile(
        const std::string&          filename, 
        const std::vector<uint8_t>& databin,  
        std::ios_base::openmode     mode = std::ios_base::out 
    );

    std::vector<uint8_t> GetBinaryData() { return ReadBinaryDataTemp; };
    size_t               GetTotalSize()  { return ReadBinaryDataSize; };
};

class FileLoaderString {
protected:
    std::string ReadStringDataTemp = {};
    size_t      ReadStringDataSize = {};
public:
    // true:success, false:failed.
    bool ReadStringFile(const std::string& filename);

    // non-chache.
    bool WriterStringFile(
        const std::string&      filename, 
        const std::string&      databin, 
        std::ios_base::openmode mode = std::ios_base::out
    );

    std::string GetStringData() { return ReadStringDataTemp; }
    size_t      GetTotalSize()  { return ReadStringDataSize; }
};

#endif