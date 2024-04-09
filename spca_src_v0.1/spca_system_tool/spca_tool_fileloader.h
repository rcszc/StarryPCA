// ext_fileloader. [2023_12_03] RCSZ.
// 2024_03_26: ÒÆÖ² ext_fileloader => spca_tool_fileloader RCSZ.
// update: 2024_03_26.

#ifndef _SPCA_TOOL_FILELOADER_H
#define _SPCA_TOOL_FILELOADER_H
#include <string>
#include <vector>
#include <fstream>

class FileLoaderBinary {
protected:
    std::vector<uint8_t> ReadFileData = {};
    size_t               ReadFileSize = {};
public:
    // true:success, false:failed.
    bool ReadFileBinary(const std::string& filename);

    // non-chache.
    bool WriterFileBinary(
        const std::string&          filename, 
        const std::vector<uint8_t>& databin, 
        std::ios_base::openmode     mode = std::ios_base::out
    );

    std::vector<uint8_t> GetDataBinary();
    size_t               GetFileTotalSize();
};

class FileLoaderString {
protected:
    std::string ReadFileData = {};
    size_t      ReadFileSize = {};
public:
    // true:success, false:failed.
    bool ReadFileString(const std::string& filename);

    // non-chache.
    bool WriterFileString(
        const std::string&      filename, 
        const std::string&      databin, 
        std::ios_base::openmode mode = std::ios_base::out
    );

    std::string GetDataString();
    size_t      GetFileTotalSize();
};

#endif