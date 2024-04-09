// StarryPCA. Version 0.1.0 Alpha. RCSZ.
// LLCL: OpenCL 3.0
// C++17 x64 Release.
// StarryPCA 是一个轻量化以 OpenCL 为底层封装的多设备并行计算加速库.

#include <iostream>

#include "SPCA_v0.1alpha/spca_opencl.h"
#include "SPCA_v0.1alpha/spca_thread_pool.hpp"

using namespace std;

using namespace SpcaMatrixCalc;
using namespace SpcaTasks;

int main() {

	SpcaIndexMatrix<float> DemoMatIn(SPCA_TYPE_MATRIX2D);
	size_t DemoMatrixInSize[2] = { 16384, 16384 };

	DemoMatIn.IMatrixAlloc(DemoMatrixInSize[0], DemoMatrixInSize[1]);

	SpcaIndexMatrix<float> DemoConvMatA(SPCA_TYPE_MATRIX2D);
	SpcaIndexMatrix<float> DemoConvMatB(SPCA_TYPE_MATRIX2D);
	size_t DemoConvKernelASize[2] = { 32, 32 };
	size_t DemoConvKernelBSize[2] = { 32, 32 };

	DemoConvMatA.IMatrixAlloc(DemoConvKernelASize[0], DemoConvKernelASize[0]);
	DemoConvMatB.IMatrixAlloc(DemoConvKernelBSize[0], DemoConvKernelBSize[0]);

	SpcaIndexMatrix<float> DemoParamMat(SPCA_TYPE_MATRIX2D);
	size_t DemoConvParamSize[2] = { 2, 1 };

	DemoParamMat.IMatrixAlloc(DemoConvParamSize[0], DemoConvParamSize[1]);

	*DemoParamMat.IMatrixAddressing2D(0, 0) = 32;
	*DemoParamMat.IMatrixAddressing2D(1, 0) = 32;

	size_t DemoMatrixOutSize[2] = { 16384, 16384 };

	SpcaMatrix2Calc DemoCalc;

	DemoCalc.SpcaInitCalcSystem("DemoCompute.cl", "DemoMatrixCalc");

	for (const auto& Device : *DemoCalc.SpcaGetDevicesIndex())
		LOGCONS::PushLogger(LogInfo, "PrintDevice", DemoCalc.SpcaGetDeviceInfo(Device).c_str());

	DemoCalc.SpcaAllocWorkgroup(8, 8);
	DemoCalc.SpcaSetCalcDevice(0);

	DemoCalc.SpcaPushMatrixAttrib(DemoMatrixInSize[0], DemoMatrixInSize[1], InputMatrix);

	DemoCalc.SpcaPushMatrixAttrib(DemoConvKernelASize[0], DemoConvKernelASize[1], InputMatrix);
	DemoCalc.SpcaPushMatrixAttrib(DemoConvKernelBSize[0], DemoConvKernelBSize[1], InputMatrix);

	DemoCalc.SpcaPushMatrixAttrib(DemoConvParamSize[0], DemoConvParamSize[1], InputMatrix);

	DemoCalc.SpcaPushMatrixAttrib(DemoMatrixOutSize[0], DemoMatrixOutSize[1], OutputMatrix);

	DemoCalc.SpcaCreateMemoryOBJ();

	DemoCalc.SpcaPushMatrixData(DemoMatIn);
	DemoCalc.SpcaPushMatrixData(DemoConvMatA);
	DemoCalc.SpcaPushMatrixData(DemoConvMatB);
	DemoCalc.SpcaPushMatrixData(DemoParamMat);

	DemoCalc.SpcaWriteMatrixCalc(DemoMatrixInSize[0], DemoMatrixInSize[1]);

	auto DemoResultMat = DemoCalc.SpcaReadMatrixResult();

	cout << DemoCalc.SystemRunTotalTime << " ms" << endl;
	cout << DemoCalc.SystemWriteBandwidth << " mib/s" << endl;
	cout << DemoCalc.SystemReadBandwidth << " mib/s" << endl;
}