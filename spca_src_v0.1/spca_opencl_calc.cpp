// spca_opencl_calc.
#include "spca_opencl.h"

using namespace std;
using namespace LOGCONS;

namespace SpcaMatrixCalc {
	bool SpcaMatrix2Calc::SpcaInitCalcSystem(string cl_script_path, string function_name) {
		// init config opencl.
		ComputingResource.ContextBind = SpcaCreateContext(&ComputingResource.DeviceType);
		if (!ComputingResource.ContextBind) {
			PushLogger(LogError, MODULE_LABEL_OPENCL, "failed create context.");
			return false;
		}
		// create command queue.
		ComputingResource.CmdQueue = SpcaCreateCommandQueue(ComputingResource.ContextBind, ComputingResource.DeviceType);
		if (!ComputingResource.CmdQueue) {
			PushLogger(LogError, MODULE_LABEL_OPENCL, "failed create cmd_queue.");
			return false;
		}
		// create calc program.
		ComputingResource.ProgramObject = SpcaCreateProgram(ComputingResource.ContextBind, ComputingResource.DeviceType, cl_script_path);
		if (!ComputingResource.ProgramObject) {
			PushLogger(LogError, MODULE_LABEL_OPENCL, "failed create program.");
			return false;
		}
		// create kernel function.
		ComputingResource.KernelFunction = clCreateKernel(ComputingResource.ProgramObject, function_name.c_str(), NULL);
		if (!ComputingResource.KernelFunction) {
			PushLogger(LogError, MODULE_LABEL_OPENCL, "failed create kernel.");
			return false;
		}
		return true;
	}

	void SpcaMatrix2Calc::SpcaAllocWorkgroup(size_t x, size_t y) {
		if ((x > 1) && (y > 1)) {
			WorkingGroupSize[0] = x;
			WorkingGroupSize[1] = y;
		}
		else
			PushLogger(LogWarning, MODULE_LABEL_OPENCL, "set work_group number > 1.");
	}

	// set(push) input_mem_objects & output_mem_objects.
	void SpcaMatrix2Calc::SpcaPushMatrixAttrib(size_t matrix_x, size_t matrix_y, IOModeType mode) {
		SpcaOclMemoryObject MemoryObjAttribTemp = {};

		switch (mode) {
		case(InputMatrix):  { MemoryObjAttribTemp.MemoryModeType = SPCA_MEMOBJ_MODE_IN;  break; }
		case(OutputMatrix): {
			MemoryObjAttribTemp.MemoryModeType = SPCA_MEMOBJ_MODE_OUT;
			++ComputingMemOutObjNumber; break; }
		}
		MemoryObjAttribTemp.MemorySizeBytes = FLOAT_LENSIZE(matrix_x * matrix_y);
		MemoryObjAttribTemp.MatrixWidth     = matrix_x;
		MemoryObjAttribTemp.MatrixHeight    = matrix_y;
		
		if (MemoryObjAttribTemp.MemorySizeBytes == NULL)
			LOGCONS::PushLogger(LogWarning, MODULE_LABEL_OPENCL, "push(attrib) matrix_attribute size = 0.");
		ComputingMemObjects.push_back(MemoryObjAttribTemp);
	}

	vector<SpcaOclDevice>* SpcaMatrix2Calc::SpcaGetDevicesIndex() {
		if (PlatformHdArray.size() > NULL)
			return &PlatformHdArray;
		return nullptr;
	}

	string SpcaMatrix2Calc::SpcaGetDeviceInfo(SpcaOclDevice device) {
		// info data =fmt=> string.
		return GetDeviceInfoStr(device);
	}

	void SpcaMatrix2Calc::SpcaSetCalcDevice(size_t index) {
		if (index < PlatformHdArray.size())
			index = CalcDeviceIndexType;
		else
			PushLogger(LogWarning, MODULE_LABEL_OPENCL, "set platform_device, invalid device.");
	}

	bool SpcaMatrix2Calc::SpcaCreateMemoryOBJ() {
		// create memory_objects + set kernel parameters.
		if (!SpcaCreateMemoryObjects(ComputingResource.ContextBind, ComputingMemObjects))        return false;
		if (!SpcaSetKernelFuncParameters(ComputingResource.KernelFunction, ComputingMemObjects)) return false;
		return true;
	}

	// write matrix => matrix dataset.
	bool SpcaMatrix2Calc::SpcaPushMatrixData(SpcaIndexMatrix<float>& matrix_data) {
		if (InputDatasetCount >= ComputingMemObjects.size()) {
			PushLogger(LogError, MODULE_LABEL_OPENCL, "push(dataset) count > mem_objects.");
			return false;
		}
		// error mode | size = 0.
		if (matrix_data.GetIMatrixMode() != SPCA_TYPE_MATRIX2D || 
			ComputingMemObjects[InputDatasetCount].MemorySizeBytes != matrix_data.GetIMatrixSizeBytes()
		) {
			PushLogger(LogWarning, MODULE_LABEL_OPENCL, "push(dataset) mode != 2d | in_size != attrib_size.");
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
		// start system timer.
		RunCalcTimer.ContextTimerStart();
		RunBandwidthTimer.ContextTimerStart();

		// write data => device(gpgpu) memory.
		if (!SpcaMemoryLoadDataset(ComputingResource.CmdQueue, ComputingMemObjects, InputDataset, WriteDatasetSizeBytes)) {
			// err: mem_object == null || matrix.mode != 2d || matrix.data == null.
			PushLogger(LogError, MODULE_LABEL_OPENCL, "failed write calc_device dataset.");
			return false;
		}
		// free cache data.
		InputDataset.clear();
		InputDataset.shrink_to_fit();

		// calc write speed, size > 128mib.
		double SizeMiB = double(WriteDatasetSizeBytes) / 1048576.0;
		if (SizeMiB > 128.0)
			SystemWriteBandwidth = SizeMiB / RunBandwidthTimer.ContextTimerEnd() * 1000.0;

		size_t MatrixNumber[2] = { global_size_x, global_size_y };
		// [OpenCL API]: Task => Queue, CALC(2D).
		int32_t OCLerrorCode = SpcaCLEnqueueNDRangeKernel(
			ComputingResource.CmdQueue, ComputingResource.KernelFunction,
			2, NULL, MatrixNumber, WorkingGroupSize, 
			NULL, nullptr, nullptr
		);
		// opencl failed execution.
		if (OCLerrorCode != CL_SUCCESS) {
			// write execution error.
			PushLogger(LogError, MODULE_LABEL_OPENCL, "push(add) execution_queue, code: %i", OCLerrorCode);
			// err => free resources.
			SYSSPCA_FREE_GROUP(ComputingResource);
			return false;
		}
		// clear count.
		InputDatasetCount = NULL;
		return true;
	}

	// gpu memory => data(host).
	vector<SpcaIndexMatrix<float>> SpcaMatrix2Calc::SpcaReadMatrixResult() {
		size_t WriteDatasetSizeBytes = NULL;
		vector<SpcaIndexMatrix<float>> ReturnMatrix(ComputingMemOutObjNumber, SpcaIndexMatrix<float>(SPCA_TYPE_MATRIX2D));
		
		RunBandwidthTimer.ContextTimerStart();
		if (!SpcaMemoryReadDataset(ComputingResource.CmdQueue, ComputingMemObjects, ReturnMatrix, WriteDatasetSizeBytes)) {
			// err: mem_object == null.
			PushLogger(LogError, MODULE_LABEL_OPENCL, "failed read calc_device dataset.");

			ReturnMatrix.clear();
			ReturnMatrix.shrink_to_fit();
			return ReturnMatrix;
		}

		// calc read speed, size > 128mib.
		double SizeMiB = double(WriteDatasetSizeBytes) / 1048576.0;
		if (SizeMiB > 128.0)
			SystemReadBandwidth = SizeMiB / RunBandwidthTimer.ContextTimerEnd() * 1000.0;

		// kernel calc timer(end).
		SystemRunTotalTime = RunCalcTimer.ContextTimerEnd();
		return ReturnMatrix;
	}
}