// spca_tool_matrix, (index_matrix), v0.1, RCSZ 2024.03.24
// update: 2024.03.26

#ifndef _SPCA_TOOL_MATRIX_H
#define _SPCA_TOOL_MATRIX_H
#include <vector>

// float: max:4gib.
#define SPCA_SYS_MATRIX_MAXSIZE (size_t)1073741824

#define SPCA_MATRIX_FAILED  0
#define SPCA_MATRIX_SUCCESS 1

using SpcaMatrixMode = uint32_t;

#define SPCA_TYPE_MATRIX1D (SpcaMatrixMode)1 << 1 // matrix dim:1d
#define SPCA_TYPE_MATRIX2D (SpcaMatrixMode)1 << 2 // matrix dim:2d
#define SPCA_TYPE_MATRIX3D (SpcaMatrixMode)1 << 3 // matrix dim:3d

// index_matrix dim convert [æÿ’ÛΩµŒ¨].
// original mode => convert(new) mode.
template<typename SpcaDataType>
class SpcaDimConvertREDU {
protected:
	SpcaMatrixMode IndexMatrixCvtMode = SPCA_TYPE_MATRIX1D;

	void CVTmatrix3Dto2D(size_t* matrix_dim, std::vector<SpcaDataType>& matrix_src) {
		// cvt: [x,y,z] => [x,(y * z)].
		if (matrix_src.size() == matrix_dim[0] * matrix_dim[1] * matrix_dim[2]) {
			matrix_dim[1] = matrix_dim[1] * matrix_dim[2];
			matrix_dim[2] = NULL;
		}
	}
	void CVTmatrix2Dto1D(size_t* matrix_dim, std::vector<SpcaDataType>& matrix_src) {
		// cvt: [x,y] => [x * y].
		if (matrix_src.size() == matrix_dim[0] * matrix_dim[1]) {
			matrix_dim[0] = matrix_dim[0] * matrix_dim[1];
			matrix_dim[1] = NULL;
		}
	}
	void CVTmatrix3Dto1D(size_t* matrix_dim, std::vector<SpcaDataType>& matrix_src) {
		// cvt: [x,y,z] => [x * y * z].
		if (matrix_src.size() == matrix_dim[0] * matrix_dim[1] * matrix_dim[2]) {
			matrix_dim[0] = matrix_dim[0] * matrix_dim[1] * matrix_dim[2];
			matrix_dim[1] = NULL;
			matrix_dim[2] = NULL;
		}
	}

public:
	SpcaDimConvertREDU(SpcaMatrixMode cvtmode) : IndexMatrixCvtMode(cvtmode) {}

	bool __SysConvertDimension(SpcaMatrixMode& matrix_mode, size_t* matrix_dim, std::vector<SpcaDataType>& matrix_src) {
		if (matrix_mode > IndexMatrixCvtMode) {
			// convert: 3d=>2d, 2d=>1d, 3d=>1d.
			if (matrix_mode == SPCA_TYPE_MATRIX3D && IndexMatrixCvtMode == SPCA_TYPE_MATRIX2D) CVTmatrix3Dto2D(matrix_dim, matrix_src);
			if (matrix_mode == SPCA_TYPE_MATRIX2D && IndexMatrixCvtMode == SPCA_TYPE_MATRIX1D) CVTmatrix2Dto1D(matrix_dim, matrix_src);
			if (matrix_mode == SPCA_TYPE_MATRIX3D && IndexMatrixCvtMode == SPCA_TYPE_MATRIX1D) CVTmatrix3Dto1D(matrix_dim, matrix_src);

			matrix_mode = IndexMatrixCvtMode;
			return true;
		}
		return false;
	}
};

// index_matrix dim convert [æÿ’Û…˝Œ¨].
// original mode => convert(new) mode.
template<typename SpcaDataType>
class SpcaDimConvertINCR {
protected:
	SpcaMatrixMode IndexMatrixCvtMode = SPCA_TYPE_MATRIX1D;
	size_t DimCvtTemp[3] = {};

	void CVTmatrix1Dto2D(size_t* matrix_dim, std::vector<SpcaDataType>& matrix_src) {
		// cvt: [x] => [x,y]. src_x >= x * y.
		if (matrix_dim[0] == DimCvtTemp[0] * DimCvtTemp[1] &&
			matrix_src.size() == matrix_dim[0]
		) {
			matrix_dim[0] = DimCvtTemp[0];
			matrix_dim[1] = DimCvtTemp[1];
			matrix_dim[2] = NULL;
		}
	}
	void CVTmatrix2Dto3D(size_t* matrix_dim, std::vector<SpcaDataType>& matrix_src) {
		// cvt: [x,y] => [x,y,z]. src_x = x, src_y >= y * z.
		if (matrix_dim[1] == DimCvtTemp[1] * DimCvtTemp[2] &&
			matrix_src.size() == matrix_dim[0] * matrix_dim[1]
		) {
			matrix_dim[1] = DimCvtTemp[1];
			matrix_dim[2] = DimCvtTemp[2];
		}
	}
	void CVTmatrix1Dto3D(size_t* matrix_dim, std::vector<SpcaDataType>& matrix_src) {
		// cvt: [x] => [x,y,z]. src_x >= x * y * z.
		if (matrix_dim[0] >= DimCvtTemp[0] * DimCvtTemp[1] * DimCvtTemp[2] &&
			matrix_src.size() == matrix_dim[0]
		) {
			std::memcpy(matrix_dim, DimCvtTemp, sizeof(size_t) * 3);
		}
	}

public:
	SpcaDimConvertINCR(SpcaMatrixMode cvtmode, size_t dimx, size_t dimy, size_t dimz = NULL) : 
		IndexMatrixCvtMode(cvtmode), DimCvtTemp{ dimx, dimy, dimz }
	{}

	bool __SysConvertDimension(SpcaMatrixMode& matrix_mode, size_t* matrix_dim, std::vector<SpcaDataType>& matrix_src) {
		if (matrix_mode < IndexMatrixCvtMode) {
			// convert: 1d=>2d, 2d=>3d, 1d=>3d.
			if (matrix_mode == SPCA_TYPE_MATRIX1D && IndexMatrixCvtMode == SPCA_TYPE_MATRIX2D) CVTmatrix1Dto2D(matrix_dim, matrix_src);
			if (matrix_mode == SPCA_TYPE_MATRIX2D && IndexMatrixCvtMode == SPCA_TYPE_MATRIX3D) CVTmatrix2Dto3D(matrix_dim, matrix_src);
			if (matrix_mode == SPCA_TYPE_MATRIX1D && IndexMatrixCvtMode == SPCA_TYPE_MATRIX3D) CVTmatrix1Dto3D(matrix_dim, matrix_src);

			matrix_mode = IndexMatrixCvtMode;
			return true;
		}
		return false;
	}
};

// index_matrix 1d,2d,3d.
template<typename SpcaDataType>
class SpcaIndexMatrix {
protected:
	std::vector<SpcaDataType> SourceDataArray = {};

	SpcaMatrixMode IndexMatrixMode = SPCA_TYPE_MATRIX1D;
	// data_dim(size_t): x,y,z.
	size_t IndexMatrixDim[3] = {};

public:
	SpcaIndexMatrix(SpcaMatrixMode matmode) : IndexMatrixMode(matmode) {}

	int IMatrixAlloc(size_t dimx, size_t dimy = NULL, size_t dimz = NULL) {
		size_t AllocLength = NULL;

		if (IndexMatrixMode == SPCA_TYPE_MATRIX1D) {
			IndexMatrixDim[0] = dimx;
			// alloc: x < mat_max_size, 1d(non_matrix).
			AllocLength = dimx;
		}

		if (IndexMatrixMode == SPCA_TYPE_MATRIX2D) {
			IndexMatrixDim[0] = dimx;
			IndexMatrixDim[1] = dimy;
			// matrix dim: { x:0,y:0 }, { x:1,y:n }.
			if (dimx == 0 && dimy > dimx) return SPCA_MATRIX_FAILED;
			// alloc: x * y < mat_max_size.
			AllocLength = dimx * dimy;
		}

		if (IndexMatrixMode == SPCA_TYPE_MATRIX3D) {
			IndexMatrixDim[0] = dimx;
			IndexMatrixDim[1] = dimy;
			IndexMatrixDim[2] = dimz;
			// matrix dim: { x:0,y:0,z:0 }, { x:1,y:1,z:n }.
			if (dimx == 0 && dimy > dimx) return SPCA_MATRIX_FAILED;
			if (dimy == 0 && dimz > dimy) return SPCA_MATRIX_FAILED;
			// alloc: x * y * z < mat_max_size.
			AllocLength = dimx * dimy * dimz;
		}

		if (AllocLength <= SPCA_SYS_MATRIX_MAXSIZE) {
			SourceDataArray.resize(AllocLength);
			return SPCA_MATRIX_SUCCESS;
		}
		return SPCA_MATRIX_FAILED;
	}

	size_t IMatrixFree() {
		size_t DataSizeBytes = SourceDataArray.size();

		// clear_size => free dataset.
		SourceDataArray.clear();
		SourceDataArray.shrink_to_fit();
		// clear dim_info.
		std::memset(IndexMatrixDim, 0, sizeof(SpcaDataType) * 3);

		return DataSizeBytes * sizeof(SpcaDataType);
	}

	SpcaDataType* IMatrixAddressing1D(size_t map_i) {
		// mode_flag * index.
		size_t AdsFlag = size_t(!(SPCA_TYPE_MATRIX1D ^ IndexMatrixMode));
		return &SourceDataArray[map_i * AdsFlag];
	}

	SpcaDataType* IMatrixAddressing2D(size_t map_i, size_t map_j) {
		// mode_flag * index, 2d address mapping => index.
		size_t AdsFlag = size_t(!(SPCA_TYPE_MATRIX2D ^ IndexMatrixMode));
		size_t AdsIndex = map_i * IndexMatrixDim[1] + map_j;

		// ads_mode(true): [address], ads_mode(false): [0]. 
		return &SourceDataArray[AdsIndex * AdsFlag];
	}

	SpcaDataType* IMatrixAddressing3D(size_t map_i, size_t map_j, size_t map_k) {
		// mode_flag * index, 3d address mapping => index.
		size_t AdsFlag = size_t(!(SPCA_TYPE_MATRIX3D ^ IndexMatrixMode));
		size_t AdsIndex = map_i * IndexMatrixDim[1] * IndexMatrixDim[2] + map_j * IndexMatrixDim[1] + map_k;

		// ads_mode(true): [address], ads_mode(false): [0]. 
		return &SourceDataArray[AdsIndex * AdsFlag];
	}

	void IMatrixFmtFill(SpcaDataType value) {
		// data format fill_value.
		std::fill_n(SourceDataArray.begin(), SourceDataArray.size(), value);
	}
	
	// æÿ’ÛΩµŒ¨.
	int IMatrixDimConvertLow(SpcaDimConvertREDU<SpcaDataType>& cvt_object) {
		// index_matrix =low=> convert dim.
		return cvt_object.__SysConvertDimension(IndexMatrixMode, IndexMatrixDim, SourceDataArray) 
			? SPCA_MATRIX_SUCCESS : SPCA_MATRIX_FAILED;
	}
	// æÿ’Û…˝Œ¨.
	int IMatrixDimConvertUp(SpcaDimConvertINCR<SpcaDataType>& cvt_object) {
		// index_matrix =up=> convert dim.
		return cvt_object.__SysConvertDimension(IndexMatrixMode, IndexMatrixDim, SourceDataArray)
			? SPCA_MATRIX_SUCCESS : SPCA_MATRIX_FAILED;
	}

	// matrix 0:x, 1:y, 2:z.
	size_t GetIMatrixDimParam(size_t dimindex) {
		// matrix dim parameters.
		if (dimindex > 2) dimindex = 2;
		return IndexMatrixDim[dimindex];
	}

	// matrix mid mode(1d,2d,3d).
	SpcaMatrixMode GetIMatrixMode() {
		return IndexMatrixMode;
	}
	// matrix total size(bytes).
	size_t GetIMatrixSizeBytes() { 
		return SourceDataArray.size() * sizeof(SpcaDataType); 
	}
	// warning: src_data pointer.
	std::vector<SpcaDataType>* GetIMatrixSrcData() { 
		return &SourceDataArray; 
	}
};

// æÿ’Û ˝æ›µ˜ ‘π§æﬂ.
namespace MatrixDebug {

	template<typename SpcaDataType>
	bool MatrixToolPrint1D(SpcaIndexMatrix<SpcaDataType>& matrix) {
		if (matrix.GetIMatrixMode() == SPCA_TYPE_MATRIX1D) {
			// iostream print.
			std::cout << "Matrix1D Debug: " << matrix.GetIMatrixDimParam(0) << std::endl;
			for (size_t i = 0; i < matrix.GetIMatrixDimParam(0); ++i) {
				std::cout << "[" << (SpcaDataType)*matrix.IMatrixAddressing1D(i) << "] ";
			}
			std::cout << std::endl;
			return true;
		}
		return false;
	}

	template<typename SpcaDataType>
	bool MatrixToolPrint2D(SpcaIndexMatrix<SpcaDataType>& matrix) {
		if (matrix.GetIMatrixMode() == SPCA_TYPE_MATRIX2D) {
			// iostream print.
			std::cout << "Matrix2D Debug: " << matrix.GetIMatrixDimParam(0) << " x " << matrix.GetIMatrixDimParam(1) << std::endl;
			for (size_t i = 0; i < matrix.GetIMatrixDimParam(0); ++i) {
				for (size_t j = 0; j < matrix.GetIMatrixDimParam(1); ++j)
					std::cout << "[" << (SpcaDataType)*matrix.IMatrixAddressing2D(i, j) << "] ";
				std::cout << std::endl;
			}
			std::cout << std::endl;
			return true;
		}
		return false;
	}
}

#endif