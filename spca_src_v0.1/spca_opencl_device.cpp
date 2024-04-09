// spca_opencl_device.
#include "spca_opencl.h"

using namespace std;
using namespace LOGCONS;

size_t OPENCL_TYPE_DEVICE::GET_DEVICE_INFO_constbuffer(const SpcaOclDevice& device) {
	size_t DeviceParam = NULL;
	if (clGetDeviceInfo(device.DeviceHandle, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(size_t), &DeviceParam, nullptr) != CL_SUCCESS)
		PushLogger(LogWarning, MODULE_LABEL_OPENCL, "failed get device_info: const_buffer.");
	return DeviceParam;
}

size_t OPENCL_TYPE_DEVICE::GET_DEVICE_INFO_constparams(const SpcaOclDevice& device) {
	size_t DeviceParam = NULL;
	if (clGetDeviceInfo(device.DeviceHandle, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(size_t), &DeviceParam, nullptr) != CL_SUCCESS)
		PushLogger(LogWarning, MODULE_LABEL_OPENCL, "failed get device_info: const_params.");
	return DeviceParam;
}

size_t OPENCL_TYPE_DEVICE::GET_DEVICE_INFO_globalmemory(const SpcaOclDevice& device) {
	size_t DeviceParam = NULL;
	if (clGetDeviceInfo(device.DeviceHandle, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(size_t), &DeviceParam, nullptr) != CL_SUCCESS)
		PushLogger(LogWarning, MODULE_LABEL_OPENCL, "failed get device_info: global_memory.");
	return DeviceParam;
}

size_t OPENCL_TYPE_DEVICE::GET_DEVICE_INFO_globalcache(const SpcaOclDevice& device) {
	size_t DeviceParam = NULL;
	if (clGetDeviceInfo(device.DeviceHandle, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(size_t), &DeviceParam, nullptr) != CL_SUCCESS)
		PushLogger(LogWarning, MODULE_LABEL_OPENCL, "failed get device_info: global_cache.");
	return DeviceParam;
}

size_t OPENCL_TYPE_DEVICE::GET_DEVICE_INFO_clockfrequency(const SpcaOclDevice& device) {
	size_t DeviceParam = NULL;
	if (clGetDeviceInfo(device.DeviceHandle, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(size_t), &DeviceParam, nullptr) != CL_SUCCESS)
		PushLogger(LogWarning, MODULE_LABEL_OPENCL, "failed get device_info: clock_frequency.");
	return DeviceParam;
}

size_t OPENCL_TYPE_DEVICE::GET_DEVICE_INFO_workgroup(const SpcaOclDevice& device) {
	size_t DeviceParam = NULL;
	if (clGetDeviceInfo(device.DeviceHandle, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &DeviceParam, nullptr) != CL_SUCCESS)
		PushLogger(LogWarning, MODULE_LABEL_OPENCL, "failed get device_info: work_group.");
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
		PushLogger(LogError, MODULE_LABEL_OPENCL, "opencl get platform_id, code: %i", OCLerrorCode);

	for (size_t i = 0; i < PlatformNumber; ++i) {

		clGetDeviceIDs(Platforms[i], CL_DEVICE_TYPE_ALL, NULL, nullptr, &DeviceNumber);
		Devices = new cl_device_id[DeviceNumber];

		int32_t OCLerrorCode = NULL;
		OCLerrorCode = clGetDeviceIDs(Platforms[i], CL_DEVICE_TYPE_ALL, DeviceNumber, Devices, nullptr);
		// get device_info error.
		if (OCLerrorCode != CL_SUCCESS)
			PushLogger(LogError, MODULE_LABEL_OPENCL, "opencl get device_id, code: %i, count: %u", OCLerrorCode, i);

		for (size_t j = 0; j < DeviceNumber; ++j) {
			// get platform,device handle.
			SpcaOclDevice DeviceTemp = {};
			cl_device_type DeviceTypeTemp = NULL;

			clGetDeviceInfo(Devices[j], CL_DEVICE_TYPE, sizeof(DeviceTypeTemp), &DeviceTypeTemp, nullptr);

			if (DeviceTypeTemp == CL_DEVICE_TYPE_CPU)         DeviceTemp.DeviceType = DEVICE_CPU;
			if (DeviceTypeTemp == CL_DEVICE_TYPE_GPU)         DeviceTemp.DeviceType = DEVICE_GPU;
			if (DeviceTypeTemp == CL_DEVICE_TYPE_ACCELERATOR) DeviceTemp.DeviceType = DEVICE_ACC;
			if (DeviceTypeTemp == CL_DEVICE_TYPE_DEFAULT)     DeviceTemp.DeviceType = DEVICE_DEFAULT;

			DeviceTemp.PlatformHandle = Platforms[i];
			DeviceTemp.DeviceHandle   = Devices[j];

			PlatformHdArray.push_back(DeviceTemp);
		}
		delete[] Devices;
	}
	delete[] Platforms;
}

#define CLCHAR_LENGTH 128
string OPENCL_TYPE_DEVICE::GetDeviceInfoStr(SpcaOclDevice device) {
	ostringstream StringTemp = {};
	cl_char* ParamCharTemp = new cl_char[CLCHAR_LENGTH];

	clGetPlatformInfo(device.PlatformHandle, CL_PLATFORM_NAME, CLCHAR_LENGTH, ParamCharTemp, nullptr);
	StringTemp << '\n' << "platform_model_name: " << ParamCharTemp << endl;

	clGetDeviceInfo(device.DeviceHandle, CL_DEVICE_NAME,    CLCHAR_LENGTH, ParamCharTemp, nullptr);
	StringTemp << "device_model_name: " << ParamCharTemp << endl;
	clGetDeviceInfo(device.DeviceHandle, CL_DEVICE_VENDOR,  CLCHAR_LENGTH, ParamCharTemp, nullptr);
	StringTemp << "device_vendor: "     << ParamCharTemp << endl;
	clGetDeviceInfo(device.DeviceHandle, CL_DEVICE_VERSION, CLCHAR_LENGTH, ParamCharTemp, nullptr);
	StringTemp << "device_version: "    << ParamCharTemp << endl;
	clGetDeviceInfo(device.DeviceHandle, CL_DEVICE_PROFILE, CLCHAR_LENGTH, ParamCharTemp, nullptr);
	StringTemp << "device_profile: "    << ParamCharTemp << endl;

	StringTemp << "const_buffer_size: "   << float(GET_DEVICE_INFO_constbuffer(device)) / float(1024.0f * 1024.0f) << " MiB" << endl;
	StringTemp << "const_param_max: "     << GET_DEVICE_INFO_constparams(device) << " Entry" << endl;
	StringTemp << "global_memory_size: "  << float(GET_DEVICE_INFO_globalmemory(device)) / float(1024.0f * 1024.0f) << " MiB" << endl;
	StringTemp << "global_cahce_size: "   << float(GET_DEVICE_INFO_globalcache(device)) / float(1024.0f * 1024.0f) << " MiB" << endl;
	StringTemp << "clock_frequency_max: " << GET_DEVICE_INFO_clockfrequency(device) << " MHz" << endl;

	size_t DeviceParam = GET_DEVICE_INFO_workgroup(device);
	int32_t RectMatrix = int32_t(sqrt(DeviceParam));
	StringTemp << "working_group_max: " << DeviceParam << " (" << RectMatrix << "x" << RectMatrix << ")" << endl;

	return StringTemp.str();
}