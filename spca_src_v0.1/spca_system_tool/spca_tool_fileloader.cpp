// ext_fileloader.
#include <chrono>
#include "spca_tool_fileloader.h"

using namespace std;

#define MODULE_FILELOADER "[FILE_LOADER]: "

bool FileLoaderBinary::ReadFileBinary(const std::string& filename) {
    ifstream ReadFile(filename, ios::binary);

    if (ReadFile.is_open()) {
        // get file size.
        ReadFile.seekg(0, ios::end);
        ReadFileSize = (size_t)ReadFile.tellg();
        ReadFile.seekg(0, ios::beg);

        // read binary data.
        ReadFileData.resize(ReadFileSize);
        ReadFile.read(reinterpret_cast<char*>(ReadFileData.data()), ReadFileSize);
        ReadFile.close();
        return true;
    }
    return false;
}

bool FileLoaderBinary::WriterFileBinary(const string& filename, const vector<uint8_t>& databin, ios_base::openmode mode) {
    ofstream WriteFile(filename, ios::binary | mode);

    if (WriteFile.is_open()) {
        // write binary data. 
        WriteFile.write(reinterpret_cast<const char*>(databin.data()), databin.size());
        WriteFile.close();
        return true;
    }
    return false;
}

vector<uint8_t> FileLoaderBinary::GetDataBinary()  { return ReadFileData; }
size_t          FileLoaderBinary::GetFileTotalSize() { return ReadFileSize; }

bool FileLoaderString::ReadFileString(const std::string& filename) {
    ifstream ReadFile(filename);

    if (ReadFile.is_open()) {
        // get file size.
        ReadFile.seekg(0, ios::end);
        ReadFileSize = (size_t)ReadFile.tellg();
        ReadFile.seekg(0, ios::beg);

        // read string data.
        string FileContent((istreambuf_iterator<char>(ReadFile)), istreambuf_iterator<char>());
        ReadFileData = FileContent;
        return true;
    }
    return false;
}

bool FileLoaderString::WriterFileString(const string& filename, const string& databin, ios_base::openmode mode) {
    fstream WriteFile(filename, mode);

    if (WriteFile.is_open()) {
        // write string data. 
        WriteFile.write(databin.data(), databin.size());
        WriteFile.close();
        return true;
    }
    return false;
}

string FileLoaderString::GetDataString()  { return ReadFileData; }
size_t FileLoaderString::GetFileTotalSize() { return ReadFileSize; }