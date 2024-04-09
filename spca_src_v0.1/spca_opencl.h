// spca_opencl. RCSZ 2024.03.09
// update: 2024.04.9

#ifndef _SPCA_OPENCL_H
#define _SPCA_OPENCL_H
// opencl version define.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <fstream>

#include "spca_system_tool/spca_tool_fileloader.h"
#include "spca_system_tool/spca_tool_logger.h"
#include "spca_system_tool/spca_tool_matrix.hpp"

#define MODULE_LABEL_OPENCL "SPCA_OPENCL"

#define FLOAT_LENSIZE(n) n * sizeof(float)
#define FLOAT_SIZELEN(n) n / sizeof(float)

#define WORKGROUP_DEFAULT 2

#define SPCA_MEMOBJ_MODE_IN  0xA1
#define SPCA_MEMOBJ_MODE_OUT 0xA2

// "clEnqueueNDRangeKernel" [thread-safe](global_mutex).
cl_int SpcaCLEnqueueNDRangeKernel(
	cl_command_queue command_queue,
	cl_kernel        kernel,
	cl_uint          work_dim,
	const size_t*    global_work_offset,
	const size_t*    global_work_size,
	const size_t*    local_work_size,
	cl_uint          num_events_in_wait_list,
	const cl_event*  event_wait_list,
	cl_event*        event
);

// opencl memory_object attrib.
struct SpcaOclMemoryObject {
	int32_t MemoryModeType;

	cl_mem MemoryObject;
	size_t MatrixWidth, MatrixHeight;
	size_t MemorySizeBytes;
};

// opencl calc_resources group.
struct SpcaOclGroup {

	cl_device_id     DeviceType;     // OCL 计算设备
	cl_context       ContextBind;    // OCL 上下文
	cl_command_queue CmdQueue;       // OCL 命令队列
	cl_program       ProgramObject;  // OCL 程序
	cl_kernel        KernelFunction; // OCL 核函数

	// opencl memory objects.
	std::vector<SpcaOclMemoryObject> MemObjects;
};
// free calc_resources.
bool SYSSPCA_FREE_GROUP(SpcaOclGroup resgroup);

// device types code.
enum DeviceType {
	DEVICE_DEFAULT = 1 << 1, // [default device]
	DEVICE_CPU     = 1 << 2, // [central processing unit]
	DEVICE_GPU     = 1 << 3, // [graphics processing unit]
	DEVICE_ACC     = 1 << 4  // [accelerator cards]
};
// opencl platform devices.
struct SpcaOclDevice {
	DeviceType DeviceType;

	cl_platform_id PlatformHandle;
	cl_device_id   DeviceHandle;

	// flag: false => failed_get_info.
	bool DeviceStatusFlag;
};

// spca timer. result,acc: [ms],[us].
class SpcaContextTimer {
protected:
	std::chrono::steady_clock::time_point StartTimePoint = {};
public:
	void   ContextTimerStart();
	double ContextTimerEnd();
};

// ******************************** Spca DeviceType. [OpenCL API] ********************************
class OPENCL_TYPE_DEVICE {
protected:
	std::vector<SpcaOclDevice> PlatformHdArray = {};
	
	// get platform device => string info.
	std::string GetDeviceInfoStr(SpcaOclDevice device);

	// get device parameters tool.
	size_t GET_DEVICE_INFO_constbuffer   (const SpcaOclDevice& device);
	size_t GET_DEVICE_INFO_constparams   (const SpcaOclDevice& device);
	size_t GET_DEVICE_INFO_globalmemory  (const SpcaOclDevice& device);
	size_t GET_DEVICE_INFO_globalcache   (const SpcaOclDevice& device);
	size_t GET_DEVICE_INFO_clockfrequency(const SpcaOclDevice& device);
	size_t GET_DEVICE_INFO_workgroup     (const SpcaOclDevice& device);
public:
	OPENCL_TYPE_DEVICE();
};

// ******************************** Spca OpenclCore. [OpenCL API] ********************************
class SPCA_CORE_OPENCL :public OPENCL_TYPE_DEVICE {
protected:
	std::string OpenCLprogramBuildLog = {};
	// calculation device index.
	size_t CalcDeviceIndexType = 0;

	std::string      SpcaReadKernelScript(const char* filename);
	cl_context       SpcaCreateContext(cl_device_id* device);
	cl_command_queue SpcaCreateCommandQueue(cl_context context, cl_device_id device);
	cl_program       SpcaCreateProgram(cl_context context, cl_device_id device, std::string str);

	// alloc gpgpu memory, set memory attribute. ( clCreateBuffer + clEnqueueWriteBuffer )
	bool SpcaCreateMemoryObjects(cl_context context, std::vector<SpcaOclMemoryObject>& mem_objects);
	// "in_data" matrix type = 2d. mem_obj mode = in.
	bool SpcaMemoryLoadDataset(
		cl_command_queue command, const std::vector<SpcaOclMemoryObject>& mem_objects, 
		std::vector<SpcaIndexMatrix<float>>& in_data, size_t& bytes
	);
	// "iout_data" matrix type = 2d. mem_obj mode = out.
	bool SpcaMemoryReadDataset(
		cl_command_queue command, const std::vector<SpcaOclMemoryObject>& mem_objects,
		std::vector<SpcaIndexMatrix<float>>& out_data, size_t& bytes
	);
	// set cl_script function in,out parameters.
	bool SpcaSetKernelFuncParameters(cl_kernel kernel, const std::vector<SpcaOclMemoryObject>& mem_objects);
};

// ******************************** Spca MatrixCalculation. [OpenCL API] ********************************
namespace SpcaMatrixCalc {
	enum IOModeType {
		InputMatrix  = 1 << 1,
		OutputMatrix = 1 << 2
	};

	class SpcaMatrix2Calc :public SPCA_CORE_OPENCL {
	protected:
		SpcaOclGroup ComputingResource = {};
		std::vector<SpcaOclMemoryObject> ComputingMemObjects = {};
		size_t ComputingMemOutObjNumber = NULL;

		std::vector<SpcaIndexMatrix<float>> InputDataset = {};
		size_t InputDatasetCount = NULL;

		SpcaContextTimer RunBandwidthTimer = {};
		SpcaContextTimer RunCalcTimer = {};

		size_t WorkingGroupSize[2] = { WORKGROUP_DEFAULT, WORKGROUP_DEFAULT };

	public:
		~SpcaMatrix2Calc() {
			SYSSPCA_FREE_GROUP(ComputingResource);
		};
		// write_buffer => compute => read_buffer.
		double SystemRunTotalTime = 0.0;

		// cpu <=> calc_device io speed, mib/s, start(size > 128mib). 
		double SystemWriteBandwidth = 0.0;
		double SystemReadBandwidth  = 0.0;

		// context => command_queue => program => kernel.
		bool SpcaInitCalcSystem(std::string cl_script_path, std::string function_name);

		// 分配计算设备工作组 matrix2d => [x,y].
		void SpcaAllocWorkgroup(size_t x, size_t y);

		// get device_list.
		std::vector<SpcaOclDevice>* SpcaGetDevicesIndex();
		// device_info =decode=> string.
		std::string SpcaGetDeviceInfo(SpcaOclDevice device);
		void SpcaSetCalcDevice(size_t index);

		// 2D[3D]: data_length = matrix.x * matrix.y (单个矩阵).
		void SpcaPushMatrixAttrib(size_t matrix_x, size_t matrix_y, IOModeType mode);
		// create(alloc) memory objects.
		bool SpcaCreateMemoryOBJ();

		bool SpcaPushMatrixData(SpcaIndexMatrix<float>& matrix_data);

		// dataset(host) =write=> gpu memory => exe_task.
		bool SpcaWriteMatrixCalc(size_t global_size_x, size_t global_size_y);
		// gpu memory =read=> dataset(host).
		std::vector<SpcaIndexMatrix<float>> SpcaReadMatrixResult();
	};

	namespace SpcaMatrixFilesys {
		// write matrix file_group: matrix3d.
		bool SpacFTmatrixFileGroupWrite(std::string group_folder, std::string group_name, SpcaIndexMatrix<float>& matrix_data);
		// read matrix file_group: matrix3d, success: return file_time_code, failed: return 0.
		size_t SpacFTmatrixFileGroupRead(std::string group_folder, std::string group_name, SpcaIndexMatrix<float>& matrix_data);
	}
}

#endif