// cdc_opencl.
#include <mutex>
#include "spca_opencl.h"

using namespace std;
using namespace LOGCONS;

#define OCL_PROGRAM_LOGLEN 10240

// free OpenCL resources.
bool OPENCL_FREE_RESHD(
	cl_context context, cl_command_queue command_queue, cl_program program, cl_kernel kernel,
	const vector<SpcaOclMemoryObject>& memory_obj
) {
	bool MemoryObjNull = true;
	if (!memory_obj.empty()) {
		for (const auto& mem_object : memory_obj)
			clReleaseMemObject(mem_object.MemoryObject);
	}
	else
		MemoryObjNull = false;

	if (command_queue) clReleaseCommandQueue (command_queue);
	if (kernel)        clReleaseKernel       (kernel);
	if (program)       clReleaseProgram      (program);
	if (context)       clReleaseContext      (context);

	return MemoryObjNull;
}

bool SYSSPCA_FREE_GROUP(SpcaOclGroup resgroup) {
	bool returnstate = OPENCL_FREE_RESHD(
		resgroup.ContextBind, resgroup.CmdQueue, resgroup.ProgramObject, resgroup.KernelFunction,
		resgroup.MemObjects
	);
	return returnstate;
}

void SysSpcaOpenCLerrorExit(int32_t exitcode, SpcaOclGroup resgroup) {
	// free_res => error exit.
	SYSSPCA_FREE_GROUP(resgroup);
	exit(exitcode);
}

mutex GLOBAL_CL_EXEFUNC_MUTEX = {};
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
) {
	unique_lock<mutex> Lock(GLOBAL_CL_EXEFUNC_MUTEX);
	return clEnqueueNDRangeKernel(
		command_queue, kernel,
		work_dim,
		global_work_offset,global_work_size,local_work_size,
		num_events_in_wait_list, event_wait_list, event
	);
}

void SpcaContextTimer::ContextTimerStart() {
	StartTimePoint = chrono::steady_clock::now();
}

double SpcaContextTimer::ContextTimerEnd() {
	// start - end.
	int64_t ResultMicroseconds =
		chrono::duration_cast<chrono::microseconds>(
			chrono::steady_clock::now() - StartTimePoint
		).count();

	return double(ResultMicroseconds / 1000.0);
}

// 创建 OpenCL 上下文.
cl_context SPCA_CORE_OPENCL::SpcaCreateContext(cl_device_id* device) {
	int32_t OCLerrorCode = NULL;
	cl_context Context = nullptr;

	*device = PlatformHdArray[CalcDeviceIndexType].DeviceHandle;

	Context = clCreateContext(NULL, 1, device, NULL, NULL, &OCLerrorCode);
	if (OCLerrorCode != CL_SUCCESS) {
		// opencl context error.
		PushLogger(LogError, MODULE_LABEL_OPENCL, "create opencl context, code: %i", OCLerrorCode);
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

	CommandQueue = clCreateCommandQueueWithProperties(context, device, NULL, &OCLerrorCode);
	if (OCLerrorCode != CL_SUCCESS) {
		// opencl command_queue error.
		PushLogger(LogError, MODULE_LABEL_OPENCL, "create opencl command_queue, code: %i", OCLerrorCode);
		return nullptr;
	}
	return CommandQueue;
}

// 读取 OpenCL 核函数文件.
string SPCA_CORE_OPENCL::SpcaReadKernelScript(const char* filename) {
	string ReadFileData = {};

	ifstream ReadFile(filename);
	if (ReadFile.is_open()) {
		// get file size.
		ReadFile.seekg(0, ios::end);
		size_t ReadFileSize = (size_t)ReadFile.tellg();
		ReadFile.seekg(0, ios::beg);

		// read string data.
		string FileContent((istreambuf_iterator<char>(ReadFile)), istreambuf_iterator<char>());
		ReadFileData = FileContent;
	}
	return ReadFileData;
}

// 创建 OpenCL 程序 & ReadKernel.
cl_program SPCA_CORE_OPENCL::SpcaCreateProgram(cl_context context, cl_device_id device, string str) {
	int32_t OCLerrorCode = NULL;
	cl_program Program = nullptr;

	str = SpcaReadKernelScript(str.c_str());
	char* const Source = str.data();
	Program = clCreateProgramWithSource(context, 1, (const char**)&Source, NULL, &OCLerrorCode);

	if (OCLerrorCode != CL_SUCCESS) {
		// opencl program error.
		PushLogger(LogError, MODULE_LABEL_OPENCL, "create opencl program, code: %i", OCLerrorCode);
		return nullptr;
	}

	OCLerrorCode = clBuildProgram(Program, NULL, NULL, NULL, NULL, NULL);
	if (OCLerrorCode != CL_SUCCESS) {

		OpenCLprogramBuildLog.clear();
		OpenCLprogramBuildLog.resize(OCL_PROGRAM_LOGLEN);
		clGetProgramBuildInfo(Program, device, CL_PROGRAM_BUILD_LOG, OCL_PROGRAM_LOGLEN, OpenCLprogramBuildLog.data(), NULL);
		// delete kernel program.
		clReleaseProgram(Program);
		
		// opencl build_program error.
		PushLogger(LogError, MODULE_LABEL_OPENCL, "create opencl build_program, info: %s", 
			OpenCLprogramBuildLog.c_str()
		);
		return nullptr;
	}
	return Program;
}

// 创建 OpenCL 内存对象.
bool SPCA_CORE_OPENCL::SpcaCreateMemoryObjects(cl_context context, vector<SpcaOclMemoryObject>& mem_objects) {
	int32_t OCLerrorCode = NULL;

	auto MemoryObjectInfo = [&](const char* mode, size_t size, int32_t errorcode) {
		errorcode != CL_SUCCESS ?
			LOGCONS::PushLogger(LogError, MODULE_LABEL_OPENCL, "create opencl memory, code: %i, mode: %s", errorcode, mode) :
			LOGCONS::PushLogger(LogInfo, MODULE_LABEL_OPENCL, "create opencl memory, size: %u, mode: %s", size, mode);
	};

	if (!mem_objects.empty()) {
		// set(alloc) memory_objects.
		for (size_t i = NULL; i < mem_objects.size(); ++i) {
			if (mem_objects[i].MemoryModeType == SPCA_MEMOBJ_MODE_IN) {
				// read_only memory.
				mem_objects[i].MemoryObject = clCreateBuffer(
					context, CL_MEM_READ_ONLY, 
					mem_objects[i].MemorySizeBytes, nullptr,
					&OCLerrorCode
				);
				MemoryObjectInfo("input", mem_objects[i].MemorySizeBytes, OCLerrorCode);
			}
			
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
		return true;
	}
	return false;
}

// 加载 OpenCL 计算数据集(Host => CalcDevice).
bool SPCA_CORE_OPENCL::SpcaMemoryLoadDataset(
	cl_command_queue command, const vector<SpcaOclMemoryObject>& mem_objects,
	vector<SpcaIndexMatrix<float>>& in_data, size_t& bytes
) {
	size_t LoadDataTotalSizeBytes = NULL;
	size_t InDataCount = NULL;

	for (size_t i = NULL; i < mem_objects.size(); ++i) {
		if (mem_objects[i].MemoryModeType == SPCA_MEMOBJ_MODE_IN) {
			// memory_object != null, matrix_mode = 2d, matrix_data != empty.
			if (mem_objects[i].MemoryObject != nullptr &&
				in_data[InDataCount].GetIMatrixMode() == SPCA_TYPE_MATRIX2D &&
				!in_data[InDataCount].GetIMatrixSrcData()->empty()
			) {
				int32_t OCLerrorCode = clEnqueueWriteBuffer(
					command, mem_objects[i].MemoryObject,
					CL_TRUE, NULL,
					mem_objects[i].MemorySizeBytes,
					// (N)[2024.04.09], offset address = i * data_segment_len.
					in_data[InDataCount].GetIMatrixSrcData()->data(), // + offset_ptr
					NULL, NULL, NULL
				);
				// failed load.
				if (OCLerrorCode != CL_SUCCESS) {
					// opencl loader error.
					PushLogger(LogError, MODULE_LABEL_OPENCL, "loader opencl dataset, code: %i, (obj)count: %u",
						OCLerrorCode, i);
					return false;
				}
				// size_bytes count.
				LoadDataTotalSizeBytes += mem_objects[i].MemorySizeBytes;
			}
			else {
				PushLogger(LogError, MODULE_LABEL_OPENCL,
					"loader opencl dataset, (data)count: %i, (obj)count: %i",
					InDataCount, i);
				return false;
			}
			++InDataCount;
		}
	}
	PushLogger(LogTrace, MODULE_LABEL_OPENCL, "input dataset (total)size: %.4f mib",
		(double)LoadDataTotalSizeBytes / 1048576.0);
	bytes = LoadDataTotalSizeBytes;
	return true;
}

// 读取 OpenCL 计算数据集(CalcDevice => Host).
bool SPCA_CORE_OPENCL::SpcaMemoryReadDataset(
	cl_command_queue command, const vector<SpcaOclMemoryObject>& mem_objects,
	vector<SpcaIndexMatrix<float>>& out_data, size_t& bytes
) {
	size_t ReadDataTotalSizeBytes = NULL;
	size_t OutDataCount = NULL;

	for (size_t i = NULL; i < mem_objects.size(); ++i) {
		if (mem_objects[i].MemoryModeType == SPCA_MEMOBJ_MODE_OUT) {
			// memory_object != null, matrix_mode = 2d.
			if (mem_objects[i].MemoryObject != nullptr) {
				// reassign matrix.
				out_data[OutDataCount].IMatrixFree();
				out_data[OutDataCount].IMatrixAlloc(mem_objects[i].MatrixWidth, mem_objects[i].MatrixHeight);

				int32_t OCLerrorCode = clEnqueueReadBuffer(
					command, mem_objects[i].MemoryObject,
					CL_TRUE, NULL,
					mem_objects[i].MemorySizeBytes,
					// (N)[2024.04.09], offset address = i * data_segment_len.
					out_data[OutDataCount].GetIMatrixSrcData()->data(), // + offset_ptr
					NULL, NULL, NULL
				);
				// failed read.
				if (OCLerrorCode != CL_SUCCESS) {
					// opencl loader error.
					PushLogger(LogError, MODULE_LABEL_OPENCL, "reader opencl dataset, code: %i, (obj)count: %u",
						OCLerrorCode, i);
					return false;
				}
				// size_bytes count.
				ReadDataTotalSizeBytes += mem_objects[i].MemorySizeBytes;
			}
			else {
				PushLogger(LogError, MODULE_LABEL_OPENCL,
					"reader opencl dataset, (data)count: %i, (obj)count: %i",
					OutDataCount, i);
				return false;
			}
			++OutDataCount;
		}
	}
	PushLogger(LogTrace, MODULE_LABEL_OPENCL, "output dataset (total)size: %.4f mib",
		(double)ReadDataTotalSizeBytes / 1048576.0);
	bytes = ReadDataTotalSizeBytes;
	return true;
}

// 设置 OpenCL Kernel 参数:
// parameters: 'mem_objects' in/out 顺序.
bool SPCA_CORE_OPENCL::SpcaSetKernelFuncParameters(cl_kernel kernel, const vector<SpcaOclMemoryObject>& mem_objects) {
	string KernelParametersList = {};
	int32_t OCLerrorCode = NULL;
	
	for (size_t i = 0; i < mem_objects.size(); i++) {
		// 'input' memory_objcet => set param.
		OCLerrorCode = clSetKernelArg(kernel, (uint32_t)i, sizeof(cl_mem), &mem_objects[i].MemoryObject);
		if (OCLerrorCode != CL_SUCCESS) {
			// opencl parameters error.
			PushLogger(LogError, MODULE_LABEL_OPENCL, "loader opencl parameters, code: %i, count: %u",
				OCLerrorCode, i
			);
			return false;
		}
		mem_objects[i].MemoryModeType == SPCA_MEMOBJ_MODE_IN ?
			KernelParametersList += "In(" + to_string(i) + ") " :
			KernelParametersList += "Out(" + to_string(i) + ") ";
	}
	PushLogger(LogInfo, MODULE_LABEL_OPENCL, "loader opencl parameters, list: %s",
		KernelParametersList.c_str()
	);
	return true;
}

#define MODULE_LABEL_TOOL "FUNCTOOL"

#define GROUP_FILEEXT_BIN ".bin"
#define GROUP_FILEEXT_CFG ".matcfg"
namespace SpcaMatrixCalc {
	namespace SpcaMatrixFilesys {

		bool SpacFTmatrixFileGroupWrite(string group_folder, string group_name, SpcaIndexMatrix<float>& matrix_data) {
			// convert: float_array => bin_bytes.
			vector<uint8_t> OutBinBytes(matrix_data.GetIMatrixSizeBytes());

			for (size_t i = 0; i < matrix_data.GetIMatrixSrcData()->size(); ++i)
				memcpy(&OutBinBytes[i * sizeof(float)], &(*matrix_data.GetIMatrixSrcData())[i], sizeof(float));

			string FilepathTemp = group_folder + group_name + GROUP_FILEEXT_BIN;
			FileLoaderBinary WriteBinaryFile;
			if (!WriteBinaryFile.WriterFileBinary(FilepathTemp, OutBinBytes)) {
				// loader binary err: unable write.
				PushLogger(LogError, MODULE_LABEL_TOOL, "failed write matrix file_group, no-path(wb).");
				return false;
			}

			string MatrixCfgStrTemp =
				to_string(chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now().time_since_epoch()).count()) + " ";

			// mode: matrix3d.
			if (matrix_data.GetIMatrixMode() != SPCA_TYPE_MATRIX3D) {
				PushLogger(LogError, MODULE_LABEL_TOOL, "failed write matrix file_group, mode != 3d.");
				return false;
			}
			else {
				for (size_t i = 0; i < 3; ++i)
					MatrixCfgStrTemp += to_string(matrix_data.GetIMatrixDimParam(i)) + " ";
				// matrix data filepath.
				MatrixCfgStrTemp += '\n' + FilepathTemp;
			}

			// write config(str) file.
			FileLoaderString WriteConfigFile;
			if (!WriteConfigFile.WriterFileString(group_folder + group_name + GROUP_FILEEXT_CFG, MatrixCfgStrTemp)) {
				// loader string err: unable write.
				PushLogger(LogError, MODULE_LABEL_TOOL, "failed write matrix file_group, no-path(ws).");
				return false;
			}
			return true;
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
				if (!ReadStringFile.ReadFileString(group_folder + group_name + GROUP_FILEEXT_CFG)) {
					// loader string err: unable read.
					PushLogger(LogError, MODULE_LABEL_TOOL, "failed read matrix file_group, no-path(rs).");
					return NULL;
				}
				/* config fmt:
				* time_code dim.x dim.x Dim.z
				* matrix_srcdata_filepath
				*/
				string CfgInfoTemp = ReadStringFile.GetDataString();
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
				if (!ReadBinaryFile.ReadFileBinary(DataPathTemp)) {
					// loader binary err: unable read.
					PushLogger(LogError, MODULE_LABEL_TOOL, "failed read matrix file_group, no-path(rb).");
					return NULL;
				}

				auto BinDataTemp = ReadBinaryFile.GetDataBinary();
				// convert: bin_bytes => float_array.
				if (BinDataTemp.size() != DimParam[0] * DimParam[1] * DimParam[2] * sizeof(float)) {
					PushLogger(LogError, MODULE_LABEL_TOOL, "failed read matrix file_group, dim != data_len.");
					return NULL;
				}
				else {
					size_t FloatsLen = BinDataTemp.size() / sizeof(float);
					for (size_t i = 0; i < FloatsLen; ++i)
						memcpy(&(*matrix_data.GetIMatrixSrcData())[i], &BinDataTemp[i * sizeof(float)], sizeof(float));
				}
				return TimeCode;
			}
		}
	}
}