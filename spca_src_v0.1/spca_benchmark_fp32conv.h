// spca_benchmark_fp32conv.
#ifndef _SPCA_BENCHMARK_FP32CONV_H
#define _SPCA_BENCHMARK_FP32CONV_H
#include"spca_opencl.h"

namespace SpcaBenchmarkFP32 {
	enum BenchmarkLEVEL {
		
	};

	struct SpcaBenchmarkResult {
		// conv benchmark test, v20250203.
		// time format: sec(ns).
		float BenParamsCalculateGFLOPS = 0.0f;
		float BenParamsCalculateTime   = 0.0f;

		float BenParamsMBpsUpload   = 0.0f;
		float BenParamsMBpsDownload = 0.0f;
	};

	class SpcaBenchmarkConvFP32 :protected OPENCL_TYPE_DEVICE {
	protected:
		SpcaIndexMatrix<float> BenchmarkDataMatrix = SpcaIndexMatrix<float>(SPCA_TYPE_MATRIX2D);
		SpcaIndexMatrix<float> ConvMatrixA         = SpcaIndexMatrix<float>(SPCA_TYPE_MATRIX2D);
		SpcaIndexMatrix<float> ConvMatrixB         = SpcaIndexMatrix<float>(SPCA_TYPE_MATRIX2D);
		SpcaIndexMatrix<float> ConvMatrixParams    = SpcaIndexMatrix<float>(SPCA_TYPE_MATRIX2D);

		// max concurrent threads: 320x320 max(102400).
		size_t DataMatrixSize[2]    = { 320,320 };
		size_t ConvMatrixSize[2]    = { 5120,5120 };
		size_t ConvParamsMatSize[2] = { 2,1 };

		size_t BigMatrixSize[2] = { 20480, 20480 };

		SpcaMatrixCalc::SpcaMatrix2Calc* BenchmarkSPCA = nullptr;
		SpcaBenchmarkResult ResultMessage = {};
	public:
		SpcaBenchmarkConvFP32();

		void RunBenchmarkTestConvFP32();
		void RunBenchmarkTestBandwidth();

		SpcaBenchmarkResult GetResult() const { return ResultMessage; }
		void PrintResult(const SpcaBenchmarkResult& msg);
	};
}

#endif