// spca_tool_filesystem.
#include <chrono>
#include "spca_tool_filesystem.h"

using namespace std;

bool FileLoaderBinary::ReadBinaryFile(const std::string& filename) {
    ifstream FileRead(filename, ios::binary);

    if (FileRead.is_open()) {
        // count file_size.
        FileRead.seekg(NULL, ios::end);
        ReadBinaryDataSize = (size_t)FileRead.tellg();
        FileRead.seekg(NULL, ios::beg);

        // read binary data.
        ReadBinaryDataTemp.resize(ReadBinaryDataSize);
        FileRead.read(reinterpret_cast<char*>(ReadBinaryDataTemp.data()), ReadBinaryDataSize);
        FileRead.close();
        return true;
    }
    return false;
}

bool FileLoaderBinary::WriterBinaryFile(const string& filename, const vector<uint8_t>& databin, ios_base::openmode mode) {
    ofstream FileWrite(filename, ios::binary | mode);

    if (FileWrite.is_open()) {
        // write binary data. 
        FileWrite.write(reinterpret_cast<const char*>(databin.data()), databin.size());
        FileWrite.close();
        return true;
    }
    return false;
}

bool FileLoaderString::ReadStringFile(const std::string& filename) {
    ifstream FileRead(filename);

    if (FileRead.is_open()) {
        // count file_size.
        FileRead.seekg(NULL, ios::end);
        ReadStringDataSize = (size_t)FileRead.tellg();
        FileRead.seekg(NULL, ios::beg);

        // read string data.
        string FileContent((istreambuf_iterator<char>(FileRead)), istreambuf_iterator<char>());
        ReadStringDataTemp = FileContent;
        return true;
    }
    return false;
}

bool FileLoaderString::WriterStringFile(const string& filename, const string& databin, ios_base::openmode mode) {
    fstream FileWrite(filename, mode);

    if (FileWrite.is_open()) {
        // write string data. 
        FileWrite.write(databin.data(), databin.size());
        FileWrite.close();
        return true;
    }
    return false;
}