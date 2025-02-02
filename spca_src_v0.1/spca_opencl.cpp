// cdc_opencl.
#include <mutex>
#include "spca_opencl.h"

using namespace std;
using namespace PSAG_LOGGER;

#define OCL_PROGRAM_LOGLEN 10240

class GLOBAL_START_LOGGER {
public:
	 GLOBAL_START_LOGGER() { PSAG_LOGGER_PROCESS::StartLogProcessing("system_log/"); }
	~GLOBAL_START_LOGGER() { PSAG_LOGGER_PROCESS::FreeLogProcessing(); }
};
GLOBAL_START_LOGGER GLOBAL_OBJECT;

// free opencl device resources.
bool OPENCL_FREE_RESHD(
	cl_context context, cl_command_queue command_queue, cl_program program, cl_kernel kernel,
	const vector<SpcaDeviceMemoryObject>& memory_obj
) {
	if (memory_obj.empty()) return SPCA_STATUS_FAILED;
	// free calc resource memory objects.
	for (const auto& ObjectItem : memory_obj)
		clReleaseMemObject(ObjectItem.MemoryObject);
	
	if (command_queue) clReleaseCommandQueue(command_queue);
	if (kernel) clReleaseKernel(kernel);
	if (program) clReleaseProgram(program);
	if (context) clReleaseContext(context);

	return SPCA_STATUS_SUCCESS;
}

// free calc resource group.
bool SPCA_SYS_FREE_PROGRAM(SpcaCalcProgram resgroup) {
	bool returnstate = OPENCL_FREE_RESHD(
		resgroup.ContextBind, resgroup.CmdQueue, resgroup.ProgramObject, resgroup.KernelFunction,
		resgroup.MemObjects
	);
	return returnstate;
}
// critical error program exit.
void SPCA_SYS_PROGRAM_EXIT(int32_t exitcode, SpcaCalcProgram resgroup) {
	// free resource => error exit.
	SPCA_SYS_FREE_PROGRAM(resgroup);
	exit(exitcode);
}

mutex SpcaFuncMutexEXE = {};
cl_int SpcaCLEnqueueNDRangeKernel(
	cl_command_queue command_queue, cl_kernel kernel,
	cl_uint work_dim,
	const size_t* global_work_offset, const size_t* global_work_size, const size_t* local_work_size,
	cl_uint num_events_in_wait_list, const cl_event*  event_wait_list, cl_event* event
) {
	unique_lock<mutex> Lock(SpcaFuncMutexEXE);
	return clEnqueueNDRangeKernel(
		command_queue, kernel,
		work_dim,
		global_work_offset,global_work_size,local_work_size,
		num_events_in_wait_list, event_wait_list, event
	);
}

void SpcaContextTimer::TimerContextStart() {
	TimerStartPoint = chrono::steady_clock::now();
}

double SpcaContextTimer::TimerContextEnd() {
	// context: start - end.
	int64_t ResultMicroseconds =
		chrono::duration_cast<chrono::microseconds>(
			chrono::steady_clock::now() - TimerStartPoint
		).count();
	// context time(ms).
	return double(ResultMicroseconds / 1000.0);
}

// 创建 OpenCL 上下文.
cl_context SPCA_CORE_OPENCL::SpcaCreateContext(cl_device_id* device) {
	int32_t OCLerrorCode = NULL;
	cl_context Context = nullptr;

	*device = PlatformDevicesArray[CalcDeviceIndexCode].DeviceHandle;

	Context = clCreateContext(NULL, 1, device, NULL, NULL, &OCLerrorCode);
	if (OCLerrorCode != CL_SUCCESS) {
		// opencl context error.
		PushLogger(LogError, ModuleTagOpenCL, "create opencl context, code: %i", OCLerrorCode);
		return nullptr;
	}
	return Context;
}

// 创建 OpenCL 命令队列.
cl_command_queue SPCA_CORE_OPENCL::SpcaCreateCommandQueue(cl_context context, cl_device_id device) {
	int32_t OCLerrorCode = NULL;
	cl_command_queue CommandQueue = nullptr;
	// OpenCL 2.0 的用法:
	// CommandQueue = clCreateCommandQueue(context, device, NULL, &OCLerrorCode);
	cl_queue_properties Properties[] 
		= { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
	CommandQueue = clCreateCommandQueueWithProperties(context, device, Properties, &OCLerrorCode);
	if (OCLerrorCode != CL_SUCCESS) {
		// opencl command_queue error.
		PushLogger(LogError, ModuleTagOpenCL, "create opencl command_queue, code: %i", OCLerrorCode);
		return nullptr;
	}
	return CommandQueue;
}

// 读取 OpenCL 核函数文件.
string SPCA_CORE_OPENCL::SpcaReadKernelScript(const char* filename) {
	string ReadBinaryDataTemp = {};

	ifstream FileRead(filename);
	if (FileRead.is_open()) {
		// get file size.
		FileRead.seekg(0, ios::end);
		size_t ReadBinaryDataSize = (size_t)FileRead.tellg();
		FileRead.seekg(0, ios::beg);

		// read string data.
		string FileContent((istreambuf_iterator<char>(FileRead)), istreambuf_iterator<char>());
		ReadBinaryDataTemp = FileContent;
	}
	return ReadBinaryDataTemp;
}

// 创建 OpenCL 程序 & Read: Kernel-Script.
cl_program SPCA_CORE_OPENCL::SpcaCreateProgram(cl_context context, cl_device_id device, bool is_path, string str) {
	int32_t OCLerrorCode = NULL;
	cl_program Program = nullptr;

	if (is_path) // true: str is text content filepath.
		str = SpcaReadKernelScript(str.c_str());
	char* const Source = str.data();
	Program = clCreateProgramWithSource(context, 1, (const char**)&Source, NULL, &OCLerrorCode);

	if (OCLerrorCode != CL_SUCCESS) {
		// opencl program error.
		PushLogger(LogError, ModuleTagOpenCL, "create opencl program, code: %i", OCLerrorCode);
		return nullptr;
	}
	// build kernel program.
	OCLerrorCode = clBuildProgram(Program, NULL, NULL, NULL, NULL, NULL);
	if (OCLerrorCode == CL_SUCCESS) return Program;
	
	// cache opencl compiler log_msg.
	OpenCLprogramBuildLog.clear();
	OpenCLprogramBuildLog.resize(OCL_PROGRAM_LOGLEN);
	clGetProgramBuildInfo(
		Program, device, CL_PROGRAM_BUILD_LOG, OCL_PROGRAM_LOGLEN,
		OpenCLprogramBuildLog.data(), NULL
	);
	// delete kernel program.
	clReleaseProgram(Program);
	// opencl build_program error.
	PushLogger(LogError, ModuleTagOpenCL, "create opencl build_program, info: %s",
		OpenCLprogramBuildLog.c_str()
	);
	return nullptr;
}

// 创建 OpenCL 内存对象.
bool SPCA_CORE_OPENCL::SpcaCreateMemoryObjects(cl_context context, vector<SpcaDeviceMemoryObject>& mem_objects) {
	int32_t OCLerrorCode = NULL;

	auto MemoryObjectInfo = [&](const char* mode, size_t size, int32_t errorcode) {
		errorcode != CL_SUCCESS ?
			PushLogger(LogError, ModuleTagOpenCL, "create opencl memory, code: %i, mode: %s", errorcode, mode) :
			PushLogger(LogInfo,  ModuleTagOpenCL, "create opencl memory, size: %u, mode: %s", size, mode);
	};
	// check dataset matrix size.
	if (mem_objects.empty()) return SPCA_STATUS_FAILED;

	// device alloc memory_objects.
	for (size_t i = NULL; i < mem_objects.size(); ++i) {
		// create opencl: read_only mem, spca: write_only.
		if (mem_objects[i].MemoryModeType == SPCA_MEMOBJ_MODE_IN) {
			// read_only memory.
			mem_objects[i].MemoryObject = clCreateBuffer(
				context, CL_MEM_READ_ONLY,
				mem_objects[i].MemorySizeBytes, nullptr,
				&OCLerrorCode
			);
			MemoryObjectInfo("input", mem_objects[i].MemorySizeBytes, OCLerrorCode);
		}
		// create opencl: read_write mem, spca: read_only.
		if (mem_objects[i].MemoryModeType == SPCA_MEMOBJ_MODE_OUT) {
			// read_write memory.
			mem_objects[i].MemoryObject = clCreateBuffer(
				context, CL_MEM_READ_WRITE,
				mem_objects[i].MemorySizeBytes, nullptr,
				&OCLerrorCode
			);
			MemoryObjectInfo("output", mem_objects[i].MemorySizeBytes, OCLerrorCode);
		}
	}
	return SPCA_STATUS_SUCCESS;
}


// 加载 OpenCL 计算数据集[matrix] (host => calc_device).
bool SPCA_CORE_OPENCL::SpcaMemoryDatasetLoad(
	cl_command_queue command, const vector<SpcaDeviceMemoryObject>& mem_objects,
	vector<SpcaIndexMatrix<float>>& in_data, size_t& bytes,
	vector<double>* mem_times
) {
	size_t DatasetTotalSizeBytes = NULL;
	size_t InDataCount = NULL;

	for (size_t i = NULL; i < mem_objects.size(); ++i) {
		if (mem_objects[i].MemoryModeType == SPCA_MEMOBJ_MODE_IN) {
			// memory_object != null, matrix_mode = 2d, matrix_data != empty.
			if (mem_objects[i].MemoryObject == nullptr ||
				in_data[InDataCount].GetIMatrixMode() != SPCA_TYPE_MATRIX2D ||
				in_data[InDataCount].GetIMatrixRawData()->empty()
				) {
				PushLogger(LogError, ModuleTagOpenCL, "invaild mem_object, count: %u, (obj)count: %u",
					InDataCount, i);
				return SPCA_STATUS_FAILED;
			}

			cl_event MemoryEvent = nullptr;
			int32_t OCLerrorCode = clEnqueueWriteBuffer(
				command, mem_objects[i].MemoryObject,
				CL_FALSE, NULL,
				mem_objects[i].MemorySizeBytes,
				// (N)[2024.04.09], offset address = i * data_segment_len.
				in_data[InDataCount].GetIMatrixRawData()->data(), // + offset_ptr
				NULL, nullptr, &MemoryEvent
			);
			// 测量数据上传用时. (host =upload=> calc device)
			if (mem_times != nullptr) {
				// opencl events => oper time.
				cl_ulong TimeStart = NULL, TimeEnd = NULL;
				clWaitForEvents(1, &MemoryEvent);

				clGetEventProfilingInfo(MemoryEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &TimeStart, nullptr);
				clGetEventProfilingInfo(MemoryEvent, CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &TimeEnd,   nullptr);
				// calc memory oper time: ms.
				mem_times->push_back(double(TimeEnd - TimeStart) * 1e-6);
			}
			// check upload status code.
			if (OCLerrorCode != CL_SUCCESS) {
				// opencl loader error.
				PushLogger(LogError, ModuleTagOpenCL, "loader opencl dataset, code: %i, (obj)count: %u",
					OCLerrorCode, i);
				return SPCA_STATUS_FAILED;
			}
			// size_bytes count.
			DatasetTotalSizeBytes += mem_objects[i].MemorySizeBytes;
			++InDataCount;
		}
	}
	PushLogger(LogTrace, ModuleTagOpenCL, "input dataset (total)size: %.4f mib",
		(double)DatasetTotalSizeBytes / 1048576.0);
	bytes = DatasetTotalSizeBytes;
	return SPCA_STATUS_SUCCESS;
}

// 读取 OpenCL 计算数据集[matrix] (calc_device => host).
bool SPCA_CORE_OPENCL::SpcaMemoryDatasetRead(
	cl_command_queue command, const vector<SpcaDeviceMemoryObject>& mem_objects,
	vector<SpcaIndexMatrix<float>>& out_data, size_t& bytes,
	vector<double>* mem_times
) {
	size_t ReadDataTotalSizeBytes = NULL;
	size_t OutDataCount = NULL;

	for (size_t i = NULL; i < mem_objects.size(); ++i) {
		if (mem_objects[i].MemoryModeType == SPCA_MEMOBJ_MODE_OUT) {
			// memory_object != null, matrix_mode = 2d.
			if (mem_objects[i].MemoryObject == nullptr) {
				PushLogger(LogError, ModuleTagOpenCL, "reader opencl dataset, (data)count: %i, (obj)count: %i",
					OutDataCount, i);
				return SPCA_STATUS_FAILED;
			}
			// reassign matrix.
			out_data[OutDataCount].IMatrixFree();
			out_data[OutDataCount].IMatrixAlloc(mem_objects[i].MatrixWidth, mem_objects[i].MatrixHeight);

			cl_event MemoryEvent = nullptr;
			int32_t OCLerrorCode = clEnqueueReadBuffer(
				command, mem_objects[i].MemoryObject,
				CL_FALSE, NULL,
				mem_objects[i].MemorySizeBytes,
				// (N)[2024.04.09], offset address = i * data_segment_len.
				out_data[OutDataCount].GetIMatrixRawData()->data(), // + offset_ptr
				NULL, nullptr, &MemoryEvent
			);
			// 测量数据下载用时. (calc device =download=> host)
			if (mem_times != nullptr) {
				// opencl events => oper time.
				cl_ulong TimeStart = NULL, TimeEnd = NULL;
				clWaitForEvents(1, &MemoryEvent);

				clGetEventProfilingInfo(MemoryEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &TimeStart, NULL);
				clGetEventProfilingInfo(MemoryEvent, CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &TimeEnd,   NULL);
				// calc memory oper time: ms.
				mem_times->push_back(double(TimeEnd - TimeStart) * 1e-6);
			}
			// check download status code.
			if (OCLerrorCode != CL_SUCCESS) {
				// opencl loader error.
				PushLogger(LogError, ModuleTagOpenCL, "reader opencl dataset, code: %i, (obj)count: %u",
					OCLerrorCode, i);
				return SPCA_STATUS_FAILED;
			}
			// size_bytes count.
			ReadDataTotalSizeBytes += mem_objects[i].MemorySizeBytes;
			++OutDataCount;
		}
	}
	PushLogger(LogTrace, ModuleTagOpenCL, "output dataset (total)size: %.4f mib",
		(double)ReadDataTotalSizeBytes / 1048576.0);
	bytes = ReadDataTotalSizeBytes;
	return SPCA_STATUS_SUCCESS;
}

// 设置 OpenCL kernel 参数:
// parameters: 'mem_objects' in/out 顺序.
bool SPCA_CORE_OPENCL::SpcaSetKernelFuncParameters(cl_kernel kernel, const vector<SpcaDeviceMemoryObject>& mem_objects) {
	string KernelParametersList = {};
	int32_t OCLerrorCode = NULL;
	
	for (size_t i = 0; i < mem_objects.size(); i++) {
		// 'input' memory_objcet => set param.
		OCLerrorCode = clSetKernelArg(kernel, (uint32_t)i, sizeof(cl_mem), &mem_objects[i].MemoryObject);
		if (OCLerrorCode != CL_SUCCESS) {
			// opencl parameters error.
			PushLogger(LogError, ModuleTagOpenCL, "loader opencl parameters, code: %i, count: %u",
				OCLerrorCode, i
			);
			return SPCA_STATUS_FAILED;
		}
		mem_objects[i].MemoryModeType == SPCA_MEMOBJ_MODE_IN ?
			KernelParametersList += "In(" + to_string(i) + ") " :
			KernelParametersList += "Out(" + to_string(i) + ") ";
	}
	PushLogger(LogInfo, ModuleTagOpenCL, "loader opencl parameters, list: %s",
		KernelParametersList.c_str()
	);
	return SPCA_STATUS_SUCCESS;
}

#define MODULE_LABEL_TOOL "FUNCTOOL"

#define GROUP_FILEEXT_BIN ".bin"
#define GROUP_FILEEXT_CFG ".matcfg"
namespace SpcaMatrixCalc {
	namespace SpcaMatrixFilesys {

		bool SpacFTmatrixFileGroupWrite(string group_folder, string group_name, SpcaIndexMatrix<float>& matrix_data) {
			// convert: float_array => bin_bytes.
			vector<uint8_t> OutBinBytes(matrix_data.GetIMatrixSizeBytes());

			for (size_t i = 0; i < matrix_data.GetIMatrixRawData()->size(); ++i)
				memcpy(&OutBinBytes[i * sizeof(float)], &(*matrix_data.GetIMatrixRawData())[i], sizeof(float));

			string FilepathTemp = group_folder + group_name + GROUP_FILEEXT_BIN;
			FileLoaderBinary WriteBinaryFile;
			if (!WriteBinaryFile.WriterBinaryFile(FilepathTemp, OutBinBytes)) {
				// loader binary err: unable write.
				PushLogger(LogError, MODULE_LABEL_TOOL, "failed write matrix file_group, no-path(wb).");
				return SPCA_STATUS_FAILED;
			}
			
			string MatrixTimeParam = to_string(chrono::duration_cast<chrono::microseconds>(
				chrono::steady_clock::now().time_since_epoch()
			).count()) + " ";

			// mode: matrix3d.
			if (matrix_data.GetIMatrixMode() != SPCA_TYPE_MATRIX3D) {
				PushLogger(LogError, MODULE_LABEL_TOOL, "failed write matrix file_group, mode != 3d.");
				return SPCA_STATUS_FAILED;
			}
			else {
				for (size_t i = 0; i < 3; ++i)
					MatrixTimeParam += to_string(matrix_data.GetIMatrixDimParam(i)) + " ";
				// matrix data filepath.
				MatrixTimeParam += '\n' + FilepathTemp;
			}

			// write config(str) file.
			FileLoaderString WriteConfigFile;
			if (!WriteConfigFile.WriterStringFile(group_folder + group_name + GROUP_FILEEXT_CFG, MatrixTimeParam)) {
				// loader string err: unable write.
				PushLogger(LogError, MODULE_LABEL_TOOL, "failed write matrix file_group, no-path(ws).");
				return SPCA_STATUS_FAILED;
			}
			return SPCA_STATUS_SUCCESS;
		}

		size_t SpacFTmatrixFileGroupRead(string group_folder, string group_name, SpcaIndexMatrix<float>& matrix_data) {
			// mode: matrix3d.
			if (matrix_data.GetIMatrixMode() != SPCA_TYPE_MATRIX3D) {
				PushLogger(LogError, MODULE_LABEL_TOOL, "failed read matrix file_group, mode != 3d.");
				return NULL;
			}
			else {
				if (matrix_data.GetIMatrixSizeBytes() > NULL)
					matrix_data.IMatrixFree();

				FileLoaderString ReadStringFile;
				if (!ReadStringFile.ReadStringFile(group_folder + group_name + GROUP_FILEEXT_CFG)) {
					// loader string err: unable read.
					PushLogger(LogError, MODULE_LABEL_TOOL, "failed read matrix file_group, no-path(rs).");
					return NULL;
				}
				/* config format:
				* time_code dim.x dim.x Dim.z
				* matrix_srcdata_filepath
				*/
				string CfgInfoTemp = ReadStringFile.GetStringData();
				stringstream ISS(CfgInfoTemp);

				string DimParamTemp = {}, DataPathTemp = {};
				getline(ISS, DimParamTemp, '\n');
				getline(ISS, DataPathTemp, '\n');

				stringstream PARAM(DimParamTemp);
				size_t TimeCode = NULL, DimParam[3] = {};
				PARAM >> TimeCode >> DimParam[0] >> DimParam[1] >> DimParam[2];
				// write dim, alloc_mem.
				matrix_data.IMatrixAlloc(DimParam[0], DimParam[1], DimParam[2]);

				FileLoaderBinary ReadBinaryFile;
				if (!ReadBinaryFile.ReadBinaryFile(DataPathTemp)) {
					// loader binary err: unable read.
					PushLogger(LogError, MODULE_LABEL_TOOL, "failed read matrix file_group, no-path(rb).");
					return NULL;
				}

				auto BinDataTemp = ReadBinaryFile.GetBinaryData();
				// convert: bin_bytes => float_array.
				if (BinDataTemp.size() != DimParam[0] * DimParam[1] * DimParam[2] * sizeof(float)) {
					PushLogger(LogError, MODULE_LABEL_TOOL, "failed read matrix file_group, dim != data_len.");
					return NULL;
				}
				else {
					size_t FloatsLen = BinDataTemp.size() / sizeof(float);
					for (size_t i = 0; i < FloatsLen; ++i)
						memcpy(&(*matrix_data.GetIMatrixRawData())[i], &BinDataTemp[i * sizeof(float)], sizeof(float));
				}
				return TimeCode;
			}
		}
	}
}