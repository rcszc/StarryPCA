// spca_opencl_device.
#include "spca_opencl.h"

using namespace std;
using namespace PSAG_LOGGER;

size_t OPENCL_TYPE_DEVICE::GET_DEVICE_INFO_constbuffer(const SpcaCalcDevice& device) {
	size_t DeviceParam = NULL;
	cl_int F = clGetDeviceInfo(device.DeviceHandle, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(size_t), &DeviceParam, nullptr);
	if (F != CL_SUCCESS) PushLogger(LogWarning, ModuleTagDevice, "failed get device_info: const_buffer.");
	return DeviceParam;
}

size_t OPENCL_TYPE_DEVICE::GET_DEVICE_INFO_constparams(const SpcaCalcDevice& device) {
	size_t DeviceParam = NULL;
	cl_int F = clGetDeviceInfo(device.DeviceHandle, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(size_t), &DeviceParam, nullptr);
	if (F != CL_SUCCESS) PushLogger(LogWarning, ModuleTagDevice, "failed get device_info: const_params.");
	return DeviceParam;
}

size_t OPENCL_TYPE_DEVICE::GET_DEVICE_INFO_globalmemory(const SpcaCalcDevice& device) {
	size_t DeviceParam = NULL;
	cl_int F = clGetDeviceInfo(device.DeviceHandle, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(size_t), &DeviceParam, nullptr);
	if (F != CL_SUCCESS) PushLogger(LogWarning, ModuleTagDevice, "failed get device_info: global_memory.");
	return DeviceParam;
}

size_t OPENCL_TYPE_DEVICE::GET_DEVICE_INFO_globalcache(const SpcaCalcDevice& device) {
	size_t DeviceParam = NULL;
	cl_int F = clGetDeviceInfo(device.DeviceHandle, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(size_t), &DeviceParam, nullptr);
	if (F != CL_SUCCESS) PushLogger(LogWarning, ModuleTagDevice, "failed get device_info: global_cache.");
	return DeviceParam;
}

size_t OPENCL_TYPE_DEVICE::GET_DEVICE_INFO_clockfrequency(const SpcaCalcDevice& device) {
	size_t DeviceParam = NULL;
	cl_int F = clGetDeviceInfo(device.DeviceHandle, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(size_t), &DeviceParam, nullptr);
	if (F != CL_SUCCESS) PushLogger(LogWarning, ModuleTagDevice, "failed get device_info: clock_frequency.");
	return DeviceParam;
}

size_t OPENCL_TYPE_DEVICE::GET_DEVICE_INFO_workgroup(const SpcaCalcDevice& device) {
	size_t DeviceParam = NULL;
	cl_int F = clGetDeviceInfo(device.DeviceHandle, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &DeviceParam, nullptr);
	if (F != CL_SUCCESS) PushLogger(LogWarning, ModuleTagDevice, "failed get device_info: work_group.");
	return DeviceParam;
}

OPENCL_TYPE_DEVICE::OPENCL_TYPE_DEVICE() {
	// platform,device ptr.
	cl_platform_id* Platforms = nullptr; // 平台列表.
	cl_device_id*   Devices   = nullptr; // 设备列表.
	
	cl_uint PlatformNumber = NULL;
	cl_uint DeviceNumber   = NULL;

	clGetPlatformIDs(NULL, nullptr, &PlatformNumber);
	Platforms = new cl_platform_id[PlatformNumber];

	int32_t OCLerrorCode = NULL;
	OCLerrorCode = clGetPlatformIDs(PlatformNumber, Platforms, nullptr);
	// get platform_info error.
	if (OCLerrorCode != CL_SUCCESS)
		PushLogger(LogError, ModuleTagDevice, "opencl get platform_id, code: %i", OCLerrorCode);

	for (size_t i = 0; i < PlatformNumber; ++i) {
		clGetDeviceIDs(Platforms[i], CL_DEVICE_TYPE_ALL, NULL, nullptr, &DeviceNumber);
		Devices = new cl_device_id[DeviceNumber];

		int32_t OCLerrorCode = NULL;
		OCLerrorCode = clGetDeviceIDs(Platforms[i], CL_DEVICE_TYPE_ALL, DeviceNumber, Devices, nullptr);
		// get device_info error.
		if (OCLerrorCode != CL_SUCCESS)
			PushLogger(LogError, ModuleTagDevice, "opencl get device_id, code: %i, count: %u", OCLerrorCode, i);

		for (size_t j = 0; j < DeviceNumber; ++j) {
			// get platform,device handle.
			SpcaCalcDevice DeviceTemp = {};
			cl_device_type DeviceTypeTemp = NULL;

			clGetDeviceInfo(Devices[j], CL_DEVICE_TYPE, sizeof(DeviceTypeTemp), &DeviceTypeTemp, nullptr);

			DeviceTemp.PlatformHandle = Platforms[i];
			DeviceTemp.DeviceHandle = Devices[j];

			if (DeviceTypeTemp == CL_DEVICE_TYPE_CPU)         DeviceTemp.DeviceType = DEVICE_CPU;
			if (DeviceTypeTemp == CL_DEVICE_TYPE_GPU)         DeviceTemp.DeviceType = DEVICE_GPU;
			if (DeviceTypeTemp == CL_DEVICE_TYPE_ACCELERATOR) DeviceTemp.DeviceType = DEVICE_ACC;
			if (DeviceTypeTemp == CL_DEVICE_TYPE_DEFAULT)     DeviceTemp.DeviceType = DEVICE_DEFAULT;

			PlatformDevicesArray.push_back(DeviceTemp);
		}
		delete[] Devices;
	}
	delete[] Platforms;
}

#define CLCHAR_LENGTH 128
constexpr const char* L = "[DINFO]: ";
string OPENCL_TYPE_DEVICE::GetDeviceInfoString(SpcaCalcDevice device) {
	ostringstream StringTemp = {};
	cl_char* ParamCharTemp = new cl_char[CLCHAR_LENGTH];

	clGetPlatformInfo(device.PlatformHandle, CL_PLATFORM_NAME, CLCHAR_LENGTH, ParamCharTemp, nullptr);
	StringTemp << '\n' << L << "platform_model_name: " << ParamCharTemp << endl;

	clGetDeviceInfo(device.DeviceHandle, CL_DEVICE_NAME,    CLCHAR_LENGTH, ParamCharTemp, nullptr);
	StringTemp << L << "device_model_name: " << ParamCharTemp << endl;
	clGetDeviceInfo(device.DeviceHandle, CL_DEVICE_VENDOR,  CLCHAR_LENGTH, ParamCharTemp, nullptr);
	StringTemp << L << "device_vendor: "     << ParamCharTemp << endl;
	clGetDeviceInfo(device.DeviceHandle, CL_DEVICE_VERSION, CLCHAR_LENGTH, ParamCharTemp, nullptr);
	StringTemp << L <<"device_version: "    << ParamCharTemp << endl;
	clGetDeviceInfo(device.DeviceHandle, CL_DEVICE_PROFILE, CLCHAR_LENGTH, ParamCharTemp, nullptr);
	StringTemp << L << "device_profile: "    << ParamCharTemp << endl;

	StringTemp << L << "const_buffer_size: "   << float(GET_DEVICE_INFO_constbuffer(device)) / float(1024.0f * 1024.0f) << " MiB" << endl;
	StringTemp << L << "const_param_max: "     << GET_DEVICE_INFO_constparams(device) << " Entry" << endl;
	StringTemp << L << "global_memory_size: "  << float(GET_DEVICE_INFO_globalmemory(device)) / float(1024.0f * 1024.0f) << " MiB" << endl;
	StringTemp << L << "global_cahce_size: "   << float(GET_DEVICE_INFO_globalcache(device)) / float(1024.0f * 1024.0f) << " MiB" << endl;
	StringTemp << L << "clock_frequency_max: " << GET_DEVICE_INFO_clockfrequency(device) << " MHz" << endl;

	size_t DeviceParam = GET_DEVICE_INFO_workgroup(device);
	int32_t RectMatrix = int32_t(sqrt(DeviceParam));
	StringTemp << L << "working_group_max: " << DeviceParam << " (" << RectMatrix << "x" << RectMatrix << ")";

	return StringTemp.str();
}