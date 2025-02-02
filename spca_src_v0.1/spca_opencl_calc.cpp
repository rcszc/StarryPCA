// spca_opencl_calc.
#include "spca_opencl.h"

using namespace std;
using namespace PSAG_LOGGER;

namespace SpcaMatrixCalc {
	bool SpcaMatrix2Calc::SpcaInitCalcSystem(ScriptModeTYPE mode,string cl_script_path, string function_name) {
		// init config opencl.
		ComputingResource.ContextBind = SpcaCreateContext(&ComputingResource.DeviceType);
		if (!ComputingResource.ContextBind) {
			PushLogger(LogError, ModuleTagOpenCL, "failed create context.");
			return false;
		}
		// create command queue.
		ComputingResource.CmdQueue = 
			SpcaCreateCommandQueue(ComputingResource.ContextBind, ComputingResource.DeviceType);
		if (!ComputingResource.CmdQueue) {
			PushLogger(LogError, ModuleTagOpenCL, "failed create cmd_queue.");
			return false;
		}
		bool ProgramScriptFlag = false;
		if (mode == CL_KERNEL_FILEPATH) ProgramScriptFlag = true;
		// create calc program.
		ComputingResource.ProgramObject = SpcaCreateProgram(
				ComputingResource.ContextBind, 
				ComputingResource.DeviceType, 
				ProgramScriptFlag, cl_script_path
			);
		if (!ComputingResource.ProgramObject) {
			PushLogger(LogError, ModuleTagOpenCL, "failed create program.");
			return false;
		}
		// create kernel function.
		ComputingResource.KernelFunction = 
			clCreateKernel(ComputingResource.ProgramObject, function_name.c_str(), NULL);
		if (!ComputingResource.KernelFunction) {
			PushLogger(LogError, ModuleTagOpenCL, "failed create kernel.");
			return false;
		}
		return true;
	}

	void SpcaMatrix2Calc::SpcaAllocWorkgroup(size_t x, size_t y) {
		if ((x <= 1) && (y <= 1)) {
			PushLogger(LogWarning, ModuleTagOpenCL, "set work_group number > 1.");
			return;
		}
		WorkingGroupSize[0] = x;
		WorkingGroupSize[1] = y;
	}

	// set(push) input_mem_objects & output_mem_objects.
	void SpcaMatrix2Calc::SpcaPushMatrixAttribute(size_t matrix_x, size_t matrix_y, IOModeTYPE mode) {
		SpcaDeviceMemoryObject MemoryObjAttribTemp = {};

		switch (mode) {
		case(WRITE_ONLY_MATRIX):  { MemoryObjAttribTemp.MemoryModeType = SPCA_MEMOBJ_MODE_IN;  break; }
		case(READ_ONLY_MATRIX): {
			MemoryObjAttribTemp.MemoryModeType = SPCA_MEMOBJ_MODE_OUT;
			++ComputingOutMemObjCount; break; }
		}
		MemoryObjAttribTemp.MemorySizeBytes = FLOAT32_LENSIZE(matrix_x * matrix_y);
		MemoryObjAttribTemp.MatrixWidth     = matrix_x;
		MemoryObjAttribTemp.MatrixHeight    = matrix_y;
		
		if (MemoryObjAttribTemp.MemorySizeBytes == NULL)
			PushLogger(LogWarning, ModuleTagOpenCL, "push(attrib) matrix_attribute size = 0.");
		ComputingResource.MemObjects.push_back(MemoryObjAttribTemp);
	}

	vector<SpcaCalcDevice>* SpcaMatrix2Calc::SpcaGetDevicesIndex() {
		if (PlatformDevicesArray.size() > NULL)
			return &PlatformDevicesArray;
		return nullptr;
	}

	string SpcaMatrix2Calc::SpcaGetDeviceInfo(SpcaCalcDevice device) {
		// messgae data format string.
		return GetDeviceInfoString(device);
	}

	void SpcaMatrix2Calc::SpcaSetCalcDevice(size_t index) {
		if (index < PlatformDevicesArray.size()) {
			index = CalcDeviceIndexCode;
			return;
		}
		PushLogger(LogWarning, ModuleTagOpenCL, "set platform_device, invalid device.");
	}

	bool SpcaMatrix2Calc::SpcaCreateMemoryOBJ() {
		// create memory_objects + set kernel parameters.
		bool ReturnStatus =
			SpcaCreateMemoryObjects(ComputingResource.ContextBind, ComputingResource.MemObjects) &&
			SpcaSetKernelFuncParameters(ComputingResource.KernelFunction, ComputingResource.MemObjects);
		return ReturnStatus;
	}

	// write matrix => matrix dataset.
	bool SpcaMatrix2Calc::SpcaPushMatrixData(SpcaIndexMatrix<float>& matrix_data) {
		if (InputDatasetCount >= ComputingResource.MemObjects.size()) {
			PushLogger(LogError, ModuleTagOpenCL, "push(dataset) count > mem_objects.");
			return false;
		}
		// error mode | size = 0.
		if (matrix_data.GetIMatrixMode() != SPCA_TYPE_MATRIX2D || 
			ComputingResource.MemObjects[InputDatasetCount].MemorySizeBytes != matrix_data.GetIMatrixSizeBytes()
		) {
			PushLogger(LogWarning, ModuleTagOpenCL, "push(dataset) mode != 2d | in_size != attrib_size.");
			return false;
		}
		InputDataset.push_back(matrix_data);
		++InputDatasetCount;
		return true;
	}

	// global_size: opencl kernel clac_cycles.
	// data(host) => gpu memory => calc.
	bool SpcaMatrix2Calc::SpcaWriteMatrixCalc(size_t global_size_x, size_t global_size_y) {
		size_t WriteDatasetSizeBytes = NULL;

		// host upload data time.
		vector<double> MemoryOperationTime = {};
		// write data => device(gpgpu) memory.
		if (!SpcaMemoryDatasetLoad(
			ComputingResource.CmdQueue, 
			ComputingResource.MemObjects,
			InputDataset, 
			WriteDatasetSizeBytes,
			&MemoryOperationTime
		)) {
			// err: mem_object == null || matrix.mode != 2d || matrix.data == null.
			PushLogger(LogError, ModuleTagOpenCL, "failed write calc_device dataset.");
			return false;
		}
		// free cache data.
		InputDataset.clear();
		InputDataset.shrink_to_fit();

		// calc total memory time(ms).
		double MemTotalTime = 0.0;
		for (auto Time : MemoryOperationTime)
			MemTotalTime += Time;
		// calc write mem speed, size > 128mib.
		double SizeMiB = double(WriteDatasetSizeBytes) / 1048576.0;
		if (SizeMiB > 128.0) SystemWriteBandwidth = SizeMiB / MemTotalTime * 1000.0;

		size_t MatrixNumber[2] = { global_size_x, global_size_y };

		cl_event RunEvent = nullptr;
		// [OpenCL API]: Task => Queue, CALC(2D).
		int32_t OCLerrorCode = SpcaCLEnqueueNDRangeKernel(
			ComputingResource.CmdQueue, ComputingResource.KernelFunction,
			2, NULL, MatrixNumber, WorkingGroupSize, 
			NULL, nullptr, &RunEvent
		);
		clWaitForEvents(1, &RunEvent);
		// opencl events => run calc time.
		cl_ulong TimeStart = NULL, TimeEnd = NULL;
		clGetEventProfilingInfo(RunEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &TimeStart, nullptr);
		clGetEventProfilingInfo(RunEvent, CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &TimeEnd,   nullptr);
		// kernel running time(ms).
		SystemRunTotalTime = double(TimeEnd - TimeStart) * 1e-6;

		// opencl failed execution.
		if (OCLerrorCode != CL_SUCCESS) {
			// write execution error.
			PushLogger(LogError, ModuleTagOpenCL, "push(add) execution_queue, code: %i", OCLerrorCode);
			// err => free resources.
			SPCA_SYS_FREE_PROGRAM(ComputingResource);
			return false;
		}
		// clear count.
		InputDatasetCount = NULL;
		return true;
	}

	// gpu memory => data(host).
	vector<SpcaIndexMatrix<float>> SpcaMatrix2Calc::SpcaReadMatrixResult() {
		size_t WriteDatasetSizeBytes = NULL;
		// clac result dataset temp.
		vector<SpcaIndexMatrix<float>> ReturnMatrix(
			ComputingOutMemObjCount, 
			SpcaIndexMatrix<float>(SPCA_TYPE_MATRIX2D)
		);

		// host download data time.
		vector<double> MemoryOperationTime = {};
		if (!SpcaMemoryDatasetRead(
			ComputingResource.CmdQueue, 
			ComputingResource.MemObjects,
			ReturnMatrix, 
			WriteDatasetSizeBytes,
			&MemoryOperationTime
		)) {
			// err: mem_object == null.
			PushLogger(LogError, ModuleTagOpenCL, "failed read calc_device dataset.");
			// clear free cache.
			ReturnMatrix.clear();
			ReturnMatrix.shrink_to_fit();
			return ReturnMatrix;
		}
		// calc total memory time(ms).
		double MemTotalTime = 0.0;
		for (auto Time : MemoryOperationTime)
			MemTotalTime += Time;
		// calc read speed, size > 128mib.
		double SizeMiB = double(WriteDatasetSizeBytes) / 1048576.0;
		if (SizeMiB > 128.0) SystemReadBandwidth = SizeMiB / MemTotalTime * 1000.0;

		// return calc result matrix.
		return ReturnMatrix;
	}
}