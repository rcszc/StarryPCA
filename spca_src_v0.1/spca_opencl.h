// spca_opencl. RCSZ 2024_03_09
// update: 2024_04_9(1.0), 2024_09_25(2.0)

#ifndef _SPCA_OPENCL_H
#define _SPCA_OPENCL_H
// opencl version define.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <fstream>

#include "spca_system_tool/spca_tool_filesystem.h"
#include "spca_system_tool/spca_tool_logger.hpp"
#include "spca_system_tool/spca_tool_matrix.hpp"

StaticStrLABEL ModuleTagDevice    = "SPCA_DEVICE";
StaticStrLABEL ModuleTagOpenCL    = "SPCA_OPENCL";
StaticStrLABEL ModuleTagBenchmark = "SPCA_BENCHMARK";

#define FLOAT32_LENSIZE(n) n * sizeof(float)
#define FLOAT32_SIZELEN(n) n / sizeof(float)

#define WORKGROUP_DEFAULT 2

#define SPCA_MEMOBJ_MODE_IN  0xA1
#define SPCA_MEMOBJ_MODE_OUT 0xA2

#define SPCA_STATUS_INVALID -1
#define SPCA_STATUS_FAILED   0
#define SPCA_STATUS_SUCCESS  1

// opencl task enqueue func. [thread-safe](global_mutex).
cl_int SpcaCLEnqueueNDRangeKernel(
	cl_command_queue command_queue, cl_kernel kernel,
	cl_uint work_dim,
	const size_t* global_work_offset, const size_t* global_work_size, const size_t* local_work_size,
	cl_uint num_events_in_wait_list, const cl_event*  event_wait_list, cl_event* event
);

// opencl device memory_object & attribute.
struct SpcaDeviceMemoryObject {
	int32_t MemoryModeType;

	cl_mem MemoryObject;
	size_t MatrixWidth, MatrixHeight;
	size_t MemorySizeBytes;
};

// opencl calc_program resource.
struct SpcaCalcProgram {
	// opencl memory objects.
	std::vector<SpcaDeviceMemoryObject> MemObjects;

	cl_device_id     DeviceType;     // [OCL] 计算设备
	cl_context       ContextBind;    // [OCL] 上下文
	cl_command_queue CmdQueue;       // [OCL] 命令队列
	cl_program       ProgramObject;  // [OCL] 程序
	cl_kernel        KernelFunction; // [OCL] 核函数
};
// free calc_program resource.
bool SPCA_SYS_FREE_PROGRAM(SpcaCalcProgram resgroup);

// device types code.
enum DeviceType {
	DEVICE_DEFAULT = 1 << 1, // [default device]
	DEVICE_CPU     = 1 << 2, // [central processing unit]
	DEVICE_GPU     = 1 << 3, // [graphics processing unit]
	DEVICE_ACC     = 1 << 4  // [accelerator cards]
};
// opencl platform & device_type.
struct SpcaCalcDevice {
	DeviceType DeviceType;

	cl_platform_id PlatformHandle;
	cl_device_id   DeviceHandle;

	// flag: false: failed get_information.
	bool DeviceStatusFlag;
};

// spca timer. result,acc: [ms],[us].
class SpcaContextTimer {
protected:
	std::chrono::steady_clock::time_point TimerStartPoint = {};
public:
	void TimerContextStart();
	double TimerContextEnd();
};

// ******************************** SPCA device_type. [OpenCL API] ********************************
class OPENCL_TYPE_DEVICE {
protected:
	std::vector<SpcaCalcDevice> PlatformDevicesArray = {};
	// get platform device info => string info.
	std::string GetDeviceInfoString(SpcaCalcDevice device);

	// get device parameters tool.
	size_t GET_DEVICE_INFO_constbuffer   (const SpcaCalcDevice& device);
	size_t GET_DEVICE_INFO_constparams   (const SpcaCalcDevice& device);
	size_t GET_DEVICE_INFO_globalmemory  (const SpcaCalcDevice& device);
	size_t GET_DEVICE_INFO_globalcache   (const SpcaCalcDevice& device);
	size_t GET_DEVICE_INFO_clockfrequency(const SpcaCalcDevice& device);
	size_t GET_DEVICE_INFO_workgroup     (const SpcaCalcDevice& device);
public:
	OPENCL_TYPE_DEVICE();
};

// ******************************** SPCA opencl calc_core. [OpenCL API] ********************************
class SPCA_CORE_OPENCL :public OPENCL_TYPE_DEVICE {
protected:
	std::string OpenCLprogramBuildLog = {};
	// calculation device index.
	size_t CalcDeviceIndexCode = 0;

	std::string      SpcaReadKernelScript(const char* filename);
	cl_context       SpcaCreateContext(cl_device_id* device);
	cl_command_queue SpcaCreateCommandQueue(cl_context context, cl_device_id device);
	cl_program       SpcaCreateProgram(cl_context context, cl_device_id device, bool is_path, std::string str);

	// alloc gpgpu memory, set memory attribute. ( clCreateBuffer + clEnqueueWriteBuffer )
	bool SpcaCreateMemoryObjects(cl_context context, std::vector<SpcaDeviceMemoryObject>& mem_objects);
	// "in_data" matrix type = 2d. mem_obj mode = in.
	bool SpcaMemoryDatasetLoad(
		cl_command_queue command, const std::vector<SpcaDeviceMemoryObject>& mem_objects, 
		std::vector<SpcaIndexMatrix<float>>& in_data, size_t& bytes, 
		std::vector<double>* mem_times = nullptr
	);
	// "out_data" matrix type = 2d. mem_obj mode = out.
	bool SpcaMemoryDatasetRead(
		cl_command_queue command, const std::vector<SpcaDeviceMemoryObject>& mem_objects,
		std::vector<SpcaIndexMatrix<float>>& out_data, size_t& bytes, 
		std::vector<double>* mem_times = nullptr
	);
	// set (cl_script)function: in & out parameters.
	bool SpcaSetKernelFuncParameters(cl_kernel kernel, const std::vector<SpcaDeviceMemoryObject>& mem_objects);
};

// ******************************** Spca MatrixCalculation. [OpenCL API] ********************************
namespace SpcaMatrixCalc {
	// kernel script: content / path.
	enum ScriptModeTYPE {
		CL_KERNEL_FILEPATH = 1 << 1,
		CL_KERNEL_STRING   = 1 << 2
	};
	// io mode: host to device / device to host.
	enum IOModeTYPE {
		WRITE_ONLY_MATRIX = 1 << 1,
		READ_ONLY_MATRIX  = 1 << 2
	};

	class SpcaMatrix2Calc :public SPCA_CORE_OPENCL {
	protected:
		SpcaCalcProgram ComputingResource = {};
		size_t ComputingOutMemObjCount = NULL;

		std::vector<SpcaIndexMatrix<float>> InputDataset = {};
		size_t InputDatasetCount = NULL;

		size_t WorkingGroupSize[2] 
			= { WORKGROUP_DEFAULT, WORKGROUP_DEFAULT };
	public:
		~SpcaMatrix2Calc() {
			SPCA_SYS_FREE_PROGRAM(ComputingResource);
			PSAG_LOGGER::PushLogger(LogInfo, ModuleTagOpenCL, " free calc resource.");
		};
		// write_buffer => compute => read_buffer.
		double SystemRunTotalTime = 0.0;

		// cpu <=> calc_device io speed, mib/s, start(size > 128mib). 
		double SystemWriteBandwidth = 0.0;
		double SystemReadBandwidth  = 0.0;

		// context => command_queue => program => kernel.
		bool SpcaInitCalcSystem(ScriptModeTYPE mode, std::string cl_script_path, std::string function_name);

		// 分配计算设备工作组 matrix2d => [x,y].
		void SpcaAllocWorkgroup(size_t x, size_t y);

		// get device(s)_list.
		std::vector<SpcaCalcDevice>* SpcaGetDevicesIndex();
		// device_info =format=> string.
		std::string SpcaGetDeviceInfo(SpcaCalcDevice device);
		void SpcaSetCalcDevice(size_t index);

		// 有序 set matrix2d x,y,mode.
		void SpcaPushMatrixAttribute(size_t matrix_x, size_t matrix_y, IOModeTYPE mode);
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
		bool SpacFTmatrixFileGroupWrite(
			std::string group_folder, std::string group_name, SpcaIndexMatrix<float>& matrix_data
		);
		// read matrix file_group: matrix3d, success: return file_time_code, failed: return 0.
		size_t SpacFTmatrixFileGroupRead(
			std::string group_folder, std::string group_name, SpcaIndexMatrix<float>& matrix_data
		);
	}
}

#endif