# SPCA version 0.1

```C++17``` ```OpenCL3.0```

> __Update:__ 2024.04.09 RCSZ

## 使用时引入"spca_opencl.h"

### 索引矩阵使用
```cpp
template<typename SpcaDataType>
class SpcaIndexMatrix
```
> 1D,2D,3D 索引矩阵(底层都为1维).

创建矩阵对象(float为例):
```cpp
// SPCA_TYPE_MATRIX1D 1维矩阵
// SPCA_TYPE_MATRIX2D 2维矩阵
// SPCA_TYPE_MATRIX3D 3维矩阵
SpcaIndexMatrix<float> DemoMatrix(SPCA_TYPE_MATRIX2D);
```

分配矩阵内存:
```cpp
// "dimx" 第一维度 "dimy" 第二维度 "dimz" 第三维度
// return SPCA_MATRIX_FAILED / SPCA_MATRIX_SUCCESS
int IMatrixAlloc(size_t dimx, size_t dimy = NULL, size_t dimz = NULL);
```

释放矩阵内存:
```cpp
// return 释放空间大小(bytes)
size_t IMatrixFree();
```

获取矩阵索引数据指针:
```cpp
// 当矩阵为1D有效
SpcaDataType* IMatrixAddressing1D(size_t map_i);
// 当矩阵为2D有效
SpcaDataType* IMatrixAddressing2D(size_t map_i, size_t map_j);
// 当矩阵为3D有效
SpcaDataType* IMatrixAddressing3D(size_t map_i, size_t map_j, size_t map_k)
```

格式化填充矩阵数据:
```cpp
// "value" 格式化值
void IMatrixFmtFill(SpcaDataType value);
```

矩阵维度转换(降维):
- 3D to 2D
- 2D to 1D
- 3D to 1D
```cpp
// "cvt_object" 转换器对象
// return SPCA_MATRIX_FAILED / SPCA_MATRIX_SUCCESS
int IMatrixDimConvertLow(SpcaDimConvertREDU<SpcaDataType>& cvt_object);

// 例: 降为2D矩阵(原本为3D)
// x,y,z => x,y * z
SpcaDimConvertREDU<float> MatConvert(SPCA_TYPE_MATRIX2D);
IMatrixDimConvertLow(MatConvert);
```

矩阵维度转换(升维):
- 1D to 2D
- 2D to 3D
- 1D to 3D
```cpp
// "cvt_object" 转换器对象
// return SPCA_MATRIX_FAILED / SPCA_MATRIX_SUCCESS
int IMatrixDimConvertUp(SpcaDimConvertINCR<SpcaDataType>& cvt_object);

// 例: 升为3D矩阵(原本为2D)
// x:2, y:32 => x:2, y:4, z:8
SpcaDimConvertINCR<float> MatConvert(SPCA_TYPE_MATRIX3D, 2, 4, 8);
IMatrixDimConvertUp(MatConvert);
```

获取矩阵对象的一些参数:
```cpp
// "dimindex" 维度参数索引
// 0:x, 1:y, 2:z
size_t GetIMatrixDimParam(size_t dimindex);

// 获取矩阵当前类型(模式)
SpcaMatrixMode GetIMatrixMode();

// return 矩阵总大小(bytes)
size_t GetIMatrixSizeBytes();

// return 矩阵底层vector对象指针
std::vector<SpcaDataType>* GetIMatrixSrcData();
```

矩阵数据调试工具函数:
```cpp
// 控制台打印1D矩阵
// return 是否为1D矩阵
template<typename SpcaDataType>
bool MatrixDebug::MatrixToolPrint1D(SpcaIndexMatrix<SpcaDataType>& matrix);

// 控制台打印2D矩阵
// return 是否为2D矩阵
template<typename SpcaDataType>
bool MatrixDebug::MatrixToolPrint2D(SpcaIndexMatrix<SpcaDataType>& matrix);
```

---

### 调用矩阵并行计算

```cpp
class SpcaMatrixCalc::SpcaMatrix2Calc
```
> 2D矩阵并行计算, 输入输出数据均为2D矩阵.

初始化与分配计算设备工作组:
```cpp
// "cl_script_path" CL核文件路径
// "function_name"  CL核函数名称
// return status
bool SpcaInitCalcSystem(std::string cl_script_path, std::string function_name);

// "x" 二维工作组x(数量) "y" 二维工作组y(数量)
void SpcaAllocWorkgroup(size_t x, size_t y);
```

获取设备列表, 信息, 设置计算设备:
```cpp
// return 设备列表
std::vector<SpcaOclDevice>* SpcaGetDevicesIndex();

// "device" CL能识别的设备
// return 设备信息(字符串)
std::string SpcaGetDeviceInfo(SpcaOclDevice device);

// "index" 设置计算设备, 索引按照"SpcaGetDevicesIndex"成员返回的vector索引
void SpcaSetCalcDevice(size_t index);
```

配置输入输出矩阵大小(内存对象), 创建内存对象:
- 使用Push配置多个内存对象时为有序, 核参数为从左到右.
```cpp
// "matrix_x" 矩阵x(大小) "matrix_y" 矩阵y(大小)
// "mode" 矩阵模式: 输入矩阵(只读内存对象), 输出矩阵(读写内存对象)
// mode: InputMatrix / OutputMatrix
void SpcaPushMatrixAttrib(size_t matrix_x, size_t matrix_y, IOModeType mode);
		
// return status
bool SpcaCreateMemoryOBJ();
```

写入矩阵数据准备计算:
- 写入数据的顺序与以上配置内存对象顺序必须相同, 调用次数与创建的"InputMatrix"模式矩阵数量相同.
```cpp
// "matrix_data" 矩阵(2D)
// return status
bool SpcaPushMatrixData(SpcaIndexMatrix<float>& matrix_data);
```

数据写入到计算设备, 开始计算:
```cpp
// "global_size_x" CL核参数(维度执行次数): get_global_size(0)
// "global_size_y" CL核参数(维度执行次数): get_global_size(1)
// return status
bool SpcaWriteMatrixCalc(size_t global_size_x, size_t global_size_y);
```

计算完成, 获取结果(数据写回主机):
```cpp
// return N个2D矩阵(读写内存对象)
std::vector<SpcaIndexMatrix<float>> SpcaReadMatrixResult();
```

获取计算时一些设备性能参数:
```cpp
// 总共运行用时, 单位 ms
double SystemRunTotalTime;

// 写入数据带宽(主机到设备), 单位 MiB/s
double SystemWriteBandwidth;
// 读取数据带宽(设备到主机), 单位 MiB/s
double SystemReadBandwidth;
```

文件和3D矩阵操作的工具函数:
> SpcaMatrixCalc::SpcaMatrixFilesys::
```cpp
// 3D矩阵保存为文件:
// "group_folder" 保存目录 "group_name" 名称 "matrix_data" 矩阵数据
// return status
bool SpacFTmatrixFileGroupWrite(std::string group_folder, std::string group_name, SpcaIndexMatrix<float>& matrix_data);

// 读取文件3D矩阵文件:
// "group_folder" 保存目录 "group_name" 名称 "matrix_data" 矩阵数据
// return 数据大小(bytes)
size_t SpacFTmatrixFileGroupRead(string group_folder, string group_name, SpcaIndexMatrix<float>& matrix_data);
```

PS: 内部引入的OpenCL头为: ```#include <CL/cl.h>```

---

> spca_thread_pool.hpp 多线程任务使用,非常简单,可以去看看源文件就知道怎么使用了~

---

```END```