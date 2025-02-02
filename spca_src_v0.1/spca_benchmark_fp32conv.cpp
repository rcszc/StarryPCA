// spca_benchmark_fp32conv.
#include "spca_benchmark_fp32conv.h"

using namespace std;
using namespace PSAG_LOGGER;

constexpr const char* ScriptBenchmarkConvFP32 = R"(
__kernel void BenchmarkMatrixCalculate(
    __global const float* MatrixIn, 
    __global const float* ConvKernelA, __global const float* ConvKernelB,
    __global const float* ConvParam, __global float* MatrixOut
) {
    int width  = get_global_size(0);
    int height = get_global_size(1);

    int i = get_global_id(0);
    int j = get_global_id(1);

    int RangeSizeX = (int)ConvParam[0];
    int RangeSizeY = (int)ConvParam[1];

    int RangeCenterX = RangeSizeX / 2;
    int RangeCenterY = RangeSizeY / 2;

    const int blockWidth = 2;
    const int blockHeight = 2;

    float ResultValue = 0.0f;

    for (int kx = 0; kx < RangeSizeX; ++kx) {
        for (int ky = 0; ky < RangeSizeY; ++ky) {

            int InputX = i + kx - RangeCenterX;
            int InputY = j + ky - RangeCenterY;

            if (InputX >= 0 && InputX < width && InputY >= 0 && InputY < height) {
                float InputVal = MatrixIn[InputY * width + InputX];
                
                float ConvValueA = ConvKernelA[ky * RangeSizeX + kx];
                float ConvValueB = ConvKernelB[ky * RangeSizeX + kx];

                float Temp = InputVal * (ConvValueA + ConvValueB);

                // blend count 32-cycles, 32 oper + 2 oper.
                for (float idx = 0.0f; idx < 3.2f; idx += 0.1f) {
                    ResultValue += Temp * idx;
                }
            }
        }
    }
    MatrixOut[j * width + i] = ResultValue;
}
)";

constexpr const char* ScriptBenchmarkBandwidth = R"(
__kernel void BenchmarkMatrixCopy(
    __global const float* MatrixIn, __global float* MatrixOut
) {
    int width = get_global_size(0);
    
    int i = get_global_id(0);
    int j = get_global_id(1);

    MatrixOut[j * width + i] = MatrixIn[j * width + i];
}
)";

namespace SpcaBenchmarkFP32 {

    SpcaBenchmarkConvFP32::SpcaBenchmarkConvFP32() {
        BenchmarkSPCA = new SpcaMatrixCalc::SpcaMatrix2Calc();

        size_t DeviceCount = NULL;
        for (const auto& Device : *BenchmarkSPCA->SpcaGetDevicesIndex()) {
            // print devices info params.
            PushLogger(LogPerfmac, ModuleTagBenchmark, "device count %u:", DeviceCount);
            PushLogger(LogInfo,    ModuleTagBenchmark, "device information: %s",
                BenchmarkSPCA->SpcaGetDeviceInfo(Device).c_str());
            ++DeviceCount;
        }
        delete BenchmarkSPCA;
    }

    void SpcaBenchmarkConvFP32::RunBenchmarkTestConvFP32() {
        // alloc matrix host_memory space.
        BenchmarkDataMatrix.IMatrixAlloc(DataMatrixSize[0], DataMatrixSize[1]);
        ConvMatrixA.IMatrixAlloc(ConvMatrixSize[0], ConvMatrixSize[1]);
        ConvMatrixB.IMatrixAlloc(ConvMatrixSize[0], ConvMatrixSize[1]);

        ConvMatrixParams.IMatrixAlloc(ConvParamsMatSize[0], ConvParamsMatSize[1]);

        *ConvMatrixParams.IMatrixAddressing2D(0, 0) = (float)ConvMatrixSize[0];
        *ConvMatrixParams.IMatrixAddressing2D(1, 0) = (float)ConvMatrixSize[0];

        // create calc object.
        BenchmarkSPCA = new SpcaMatrixCalc::SpcaMatrix2Calc();
        BenchmarkSPCA->SpcaInitCalcSystem(
            SpcaMatrixCalc::CL_KERNEL_STRING,
            ScriptBenchmarkConvFP32, "BenchmarkMatrixCalculate"
        );
        auto DeviceInfoTemp = (*BenchmarkSPCA->SpcaGetDevicesIndex())[0];

        size_t WorkgroupTotal = GET_DEVICE_INFO_workgroup(DeviceInfoTemp);
        size_t WorkgroupDim   = size_t(sqrt(WorkgroupTotal));

        BenchmarkSPCA->SpcaAllocWorkgroup(WorkgroupDim, WorkgroupDim);
        BenchmarkSPCA->SpcaSetCalcDevice(0);

        BenchmarkSPCA->SpcaPushMatrixAttribute(DataMatrixSize[0], DataMatrixSize[1], SpcaMatrixCalc::WRITE_ONLY_MATRIX);
        BenchmarkSPCA->SpcaPushMatrixAttribute(ConvMatrixSize[0], ConvMatrixSize[1], SpcaMatrixCalc::WRITE_ONLY_MATRIX);
        BenchmarkSPCA->SpcaPushMatrixAttribute(ConvMatrixSize[0], ConvMatrixSize[1], SpcaMatrixCalc::WRITE_ONLY_MATRIX);

        BenchmarkSPCA->SpcaPushMatrixAttribute(ConvParamsMatSize[0], ConvParamsMatSize[1], SpcaMatrixCalc::WRITE_ONLY_MATRIX);
        BenchmarkSPCA->SpcaPushMatrixAttribute(DataMatrixSize[0], DataMatrixSize[1], SpcaMatrixCalc::READ_ONLY_MATRIX);

        // create opencl memory_object(s).
        BenchmarkSPCA->SpcaCreateMemoryOBJ();

        BenchmarkSPCA->SpcaPushMatrixData(BenchmarkDataMatrix);
        BenchmarkSPCA->SpcaPushMatrixData(ConvMatrixA);
        BenchmarkSPCA->SpcaPushMatrixData(ConvMatrixB);
        BenchmarkSPCA->SpcaPushMatrixData(ConvMatrixParams);

        BenchmarkSPCA->SpcaWriteMatrixCalc(DataMatrixSize[0], DataMatrixSize[1]);
        
        vector<SpcaIndexMatrix<float>> ResultMatrix = BenchmarkSPCA->SpcaReadMatrixResult();
        for (auto& Mat : ResultMatrix) Mat.IMatrixFree();

        // "BenchmarkMatrixCalculate" v20250203 RCSZ. oper = x * y * m * n * const(34).
        size_t OperFp32 = DataMatrixSize[0] * DataMatrixSize[1] * ConvMatrixSize[0] * ConvMatrixSize[1] * 34;

        ResultMessage.BenParamsCalculateGFLOPS = 
            (float)OperFp32 / 1000000000.0f / ((float)BenchmarkSPCA->SystemRunTotalTime / 1000.0f);
        ResultMessage.BenParamsCalculateTime = (float)BenchmarkSPCA->SystemRunTotalTime;

        // free test matrix memory.
        BenchmarkDataMatrix.IMatrixFree();
        ConvMatrixA.IMatrixFree();
        ConvMatrixB.IMatrixFree();
        ConvMatrixParams.IMatrixFree();

        // free clac object.
        delete BenchmarkSPCA;
    }

    void SpcaBenchmarkConvFP32::RunBenchmarkTestBandwidth() {
        // alloc matrix host_memory space.
        BenchmarkDataMatrix.IMatrixAlloc(BigMatrixSize[0], BigMatrixSize[1]);

        // create calc object.
        BenchmarkSPCA = new SpcaMatrixCalc::SpcaMatrix2Calc();
        BenchmarkSPCA->SpcaInitCalcSystem(
            SpcaMatrixCalc::CL_KERNEL_STRING,
            ScriptBenchmarkBandwidth, "BenchmarkMatrixCopy"
        );
        auto DeviceInfoTemp = (*BenchmarkSPCA->SpcaGetDevicesIndex())[0];

        size_t WorkgroupTotal = GET_DEVICE_INFO_workgroup(DeviceInfoTemp);
        size_t WorkgroupDim = size_t(sqrt(WorkgroupTotal));

        BenchmarkSPCA->SpcaAllocWorkgroup(WorkgroupDim, WorkgroupDim);
        BenchmarkSPCA->SpcaSetCalcDevice(0);

        BenchmarkSPCA->SpcaPushMatrixAttribute(BigMatrixSize[0], BigMatrixSize[1], SpcaMatrixCalc::WRITE_ONLY_MATRIX);
        BenchmarkSPCA->SpcaPushMatrixAttribute(BigMatrixSize[0], BigMatrixSize[1], SpcaMatrixCalc::READ_ONLY_MATRIX);

        // create opencl memory_object(s).
        BenchmarkSPCA->SpcaCreateMemoryOBJ();

        BenchmarkSPCA->SpcaPushMatrixData(BenchmarkDataMatrix);
        BenchmarkSPCA->SpcaWriteMatrixCalc(BigMatrixSize[0], BigMatrixSize[1]);

        vector<SpcaIndexMatrix<float>> ResultMatrix = BenchmarkSPCA->SpcaReadMatrixResult();
        for (auto& Mat : ResultMatrix) Mat.IMatrixFree();

        ResultMessage.BenParamsMBpsUpload   = (float)BenchmarkSPCA->SystemWriteBandwidth;
        ResultMessage.BenParamsMBpsDownload = (float)BenchmarkSPCA->SystemReadBandwidth;

        // free test matrix memory.
        BenchmarkDataMatrix.IMatrixFree();

        // free clac object.
        delete BenchmarkSPCA;
    }

    constexpr const char* L = "[BINFO]: ";
    void SpcaBenchmarkConvFP32::PrintResult(const SpcaBenchmarkResult& msg) {
        ostringstream StringTemp = {};
        StringTemp << endl;
        StringTemp << L << "calc total time: " << ResultMessage.BenParamsCalculateTime << " ms" << endl;
        StringTemp << L << "calc floating32: "
            << ResultMessage.BenParamsCalculateGFLOPS << " gflops, "
            << ResultMessage.BenParamsCalculateGFLOPS * 0.000976f << " tflops" << endl;
        StringTemp << L << "data bandwidth(ld): "
            << ResultMessage.BenParamsMBpsUpload << " mib/s, "
            << ResultMessage.BenParamsMBpsUpload * 0.000976f << " gib/s" << endl;
        StringTemp << L << "data bandwidth(st): "
            << ResultMessage.BenParamsMBpsDownload << " mib/s, "
            << ResultMessage.BenParamsMBpsDownload * 0.000976f << " gib/s";

        PushLogger(LogPerfmac, ModuleTagBenchmark, "benchmark results: %s", StringTemp.str().c_str());
        PushLogger(LogInfo,    ModuleTagBenchmark, "spca benchmark v20250203 by rcsz.");
    }
}