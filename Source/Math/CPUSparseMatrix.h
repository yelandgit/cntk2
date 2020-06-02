//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include <stdio.h>
#include "BaseMatrix.h"
//#include "CPUMatrix.h"
//#include "GPUMatrix.h"
//#include "GPUSparseMatrix.h"
//#include <map>
//#include <unordered_map>


namespace Microsoft { namespace MSR { namespace CNTK {

template<class> class CPUMatrix;


template <class ElemType>
class MATH_API CPUSparseMatrix : public BaseMatrix<ElemType>
{
	typedef BaseMatrix<ElemType> Base;

	mutable ElemType	value;			// for const ElemType& operator()

public:
	explicit CPUSparseMatrix(int flags=0) : Base(MatrixFormat(matrixFormatSparse|flags), CPUDEVICE) {}
	CPUSparseMatrix(size_t rows, size_t cols, size_t nnz, int flags=0) : Base(MatrixFormat(matrixFormatSparse|flags), CPUDEVICE) { Resize(rows,cols); if (nnz) Allocate(nnz); }
	CPUSparseMatrix(const CPUSparseMatrix<ElemType>& mat, bool shallow) { Assign(mat, shallow); }

	CPUSparseMatrix(const CPUSparseMatrix<ElemType>& mat) { SetValue(mat); }
	CPUSparseMatrix<ElemType>& operator=(const CPUSparseMatrix<ElemType>& mat) { if (&mat!=this) SetValue(mat); return *this; }

	CPUSparseMatrix(CPUSparseMatrix<ElemType>&& mat) { Assign(mat, true); mat.Release(); }
	CPUSparseMatrix<ElemType>& operator=(CPUSparseMatrix<ElemType>&& mat) { if (&mat!=this) { Assign(mat, true); mat.Release(); } return *this; }

public:
	void SetValue(const CPUSparseMatrix<ElemType>& mat);
	void SetValue(size_t rows, size_t cols, ElemType* p, int flags=matrixFlagNone);

	void SetDiagonalValue(ElemType v);
	void SetDiagonalValue(const CPUMatrix<ElemType>& v);
	CPUMatrix<ElemType> Diagonal() const;

	CPUSparseMatrix<ElemType> GetColumnSlice(size_t start, size_t len) const;

	CPUMatrix<ElemType> CopyToDense() const { CPUMatrix<ElemType> dm; Base::CopyToDense(dm); return dm; }
	CPUSparseMatrix<ElemType> CopyToBlock() const { CPUSparseMatrix<ElemType> dm; Base::CopyToBlock(dm); return dm; }
	void ConvertToFullBlock() { CPUSparseMatrix<ElemType> sm; CopyToFullBlock(sm); Assign(sm,true); }

	CPUSparseMatrix<ElemType> Transpose(bool hdr=false) const { CPUSparseMatrix<ElemType> sm; TransposeTo(sm); return sm; }
	CPUSparseMatrix<ElemType>& AssignTransposeOf(const CPUSparseMatrix<ElemType>& a) { if (&a!=this) Assign(a.Transpose(), true); return *this; }

///	void MaskColumnsValue(const CPUMatrix<char>& mask, ElemType val, size_t mcols);
///
///	CPUSparseMatrix<ElemType>& AssignOneHot(const CPUMatrix<ElemType>& a, vector<size_t>& shape, size_t axis);

///	CPUSparseMatrix<ElemType>& DoGatherColumnsOf(ElemType beta, const CPUMatrix<ElemType>& idx, const CPUSparseMatrix<ElemType>& a, ElemType alpha);
///	CPUSparseMatrix<ElemType>& DoScatterColumnsOf(ElemType beta, const CPUMatrix<ElemType>& idx, const CPUSparseMatrix<ElemType>& a, ElemType alpha) { NOT_IMPLEMENTED }
///
///	ElemType* Data() const;
///
///	size_t BufferSize() const { return GetSizeAllocated() * sizeof(ElemType); }
///	size_t GetNumElemAllocated() const { return GetSizeAllocated(); }

///	void SetGaussianRandomValue(ElemType /*mean*/, ElemType /*sigma*/, unsigned long /*seed*/)
///	{
///		NOT_IMPLEMENTED;
///	}
///
///	void SetMatrixFromCSCFormat(const index_t* pCol, const index_t* pRow, const ElemType* pVal, size_t nz, size_t rows, size_t numCols);	// no slice
///	void SetMatrixFromBSCFormat(const size_t* blockPos, const ElemType* pVal, size_t numBlocks, size_t rows, size_t numCols);			// no slice
///
///	// Dense * Sparse -> Dense
///	static void MultiplyAndWeightedAdd(ElemType alpha, const CPUMatrix<ElemType>& lhs, bool transposeA,
///										const CPUSparseMatrix<ElemType>& rhs, bool transposeB,
///										ElemType beta, CPUMatrix<ElemType>& c) { NOT_IMPLEMENTED }
///	// Sparse * Dense -> Dense
///	static void MultiplyAndWeightedAdd(ElemType alpha, const CPUSparseMatrix<ElemType>& lhs, bool transposeA,
///										const CPUMatrix<ElemType>& rhs, bool transposeB,
///										ElemType beta, CPUMatrix<ElemType>& c) { NOT_IMPLEMENTED }
///	// Dense * Sparse -> Sparse
///	static void MultiplyAndAdd(ElemType alpha, const CPUMatrix<ElemType>& a, bool transposeA,
///										const CPUSparseMatrix<ElemType>& b, bool transposeB,
///										CPUSparseMatrix<ElemType>& c);
///
///	static void ColumnwiseScaleAndWeightedAdd(ElemType alpha, const CPUSparseMatrix<ElemType>& a, const CPUMatrix<ElemType>& v, ElemType beta, CPUMatrix<ElemType>& c);
///
///	static void Scale(ElemType alpha, CPUSparseMatrix<ElemType>& rhs);
///	static void ScaleAndAdd(ElemType alpha, const CPUSparseMatrix<ElemType>& lhs, CPUMatrix<ElemType>& c);
///
///	static bool AreEqual(const CPUSparseMatrix<ElemType>& a, const CPUSparseMatrix<ElemType>& b, ElemType threshold = 1e-8) { NOT_IMPLEMENTED }

	// sum(vec(a).*vec(b))
///	static ElemType InnerProductOfMatrices(const CPUSparseMatrix<ElemType>& /*a*/, const CPUMatrix<ElemType>& /*b*/)
///	{
///		NOT_IMPLEMENTED;
///	}
///	static void InnerProduct(const CPUSparseMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c, bool isColWise) { NOT_IMPLEMENTED }

///	static void AddScaledDifference(ElemType /*alpha*/, const CPUSparseMatrix<ElemType>& /*a*/, const CPUMatrix<ElemType>& /*b*/, CPUMatrix<ElemType>& /*c*/,
///									bool /*bDefaultZero*/)
///	{
///		NOT_IMPLEMENTED;
///	}
///	static void AddScaledDifference(ElemType /*alpha*/, const CPUMatrix<ElemType>& /*a*/, const CPUSparseMatrix<ElemType>& /*b*/, CPUMatrix<ElemType>& /*c*/,
///									bool /*bDefaultZero*/)
///	{
///		NOT_IMPLEMENTED;
///	}
///
///	int GetDeviceId() const
///	{
///		return -1;
///	}

	// Allocate actually allocates the storage space for nnz elements. This is different than resizing, which changes the dimensions of the underlying matrix.
	// Unfortunately numRows/numCols need to be passed in the case of various matrix formats (e.g., SparseCSC), because some of the dimensions allocated depend on the
	// dimensions of the matrix.
	//void Allocate(size_t rows, size_t cols, size_t nnz = 10000, bool grow = true, bool keepval = false); // matrix format will affect the size to allocate
	// RequireSizeAndAllocate is required by SpasreMatrix since resizing the dimensions and allocating storage are different operations. Since a Resize can entail changing
	// MatrixFormat, we offer an overload for that case.
	//void RequireSizeAndAllocate(size_t rows, size_t cols, size_t nnz, MatrixFormat mft, bool grow = true, bool keepval = true); // matrix format will affect the size to allocate
	// Otherwise we will just use the current MatrixFormat.
	//void RequireSizeAndAllocate(size_t rows, size_t cols, size_t nnz = 10000, bool grow = true, bool keepval = false);
	// Sparse matrix RequireSize is similar to dense matrix RequireSize in that it will only allocate the minimum amount of storage required to successfully create the matrix.
	// This is required because some formats (e.g., SparseCSC) require the SecondIndexLocation to have valid data in order to compute m_nz. Otherwise this method would not
	// update the storage at all.
	//void RequireSize(size_t rows, size_t cols, size_t nnz, MatrixFormat mft, bool grow = true);
	//void RequireSize(size_t rows, size_t cols, MatrixFormat mft, bool grow = true) { return RequireSize(numRows, cols, 0, mft, grow); }
	//void RequireSize(size_t rows, size_t cols, bool grow = true);
	// Resizes the dimensions of the underlying sparse matrix object. Since the caller may have a hint for m_nz, we allow that to be passed, but this is terrible design. In the
	// future if we want better separation between allocation and resizing, this interface should be updated to be less clunky.
	//void Resize(size_t rows, size_t cols, size_t nnz, MatrixFormat mft, bool grow = true); // matrix format will affect the size to allocate
	void Resize(size_t rows, size_t cols, size_t nnz=0, bool grow = true) { Base::Resize(rows,cols); if (nnz>0) Base::Allocate(nnz); }
	//using Base::Resize;

	//void Reset();

	const ElemType& operator()(size_t row, size_t col) const { value = m_sob->GetItem(row, col, m_sliceOffset); return value; }
	//ElemType& operator()(size_t row, size_t col) { return m_sob->Item(row, col, m_sliceOffset); }

public:
///	void NormalGrad(CPUMatrix<ElemType>& c, ElemType momentum, ElemType unitGainFactor) { NOT_IMPLEMENTED }
///	ElemType Adagrad(CPUMatrix<ElemType>& c, bool needAveMultiplier) { NOT_IMPLEMENTED }
///
///	template<typename AccumType>
///	void AdaDelta(CPUMatrix<AccumType>& c, CPUMatrix<AccumType>& functionValues, AccumType learningRate, AccumType rho, AccumType epsilon, int* timestamps, int currentTimestamp) { NOT_IMPLEMENTED }

public:
///	CPUSparseMatrix<ElemType>& InplaceTruncateTop(ElemType threshold) { NOT_IMPLEMENTED }
///	CPUSparseMatrix<ElemType>& InplaceTruncateBottom(ElemType threshold) { NOT_IMPLEMENTED }
///	CPUSparseMatrix<ElemType>& InplaceTruncate(ElemType threshold) { NOT_IMPLEMENTED }
///	CPUSparseMatrix<ElemType>& InplaceSoftThreshold(ElemType threshold) { NOT_IMPLEMENTED }
///
///	ElemType FrobeniusNorm() const { NOT_IMPLEMENTED }
///
///	ElemType SumOfAbsElements() const; // sum of all abs(elements)
///	ElemType SumOfElements() const { NOT_IMPLEMENTED }	// sum of all elements

public:
///	void Print(const char* matrixName, ptrdiff_t rowStart, ptrdiff_t rowEnd, ptrdiff_t colStart, ptrdiff_t colEnd) const;
///	void Print(const char* matrixName = NULL) const; // print whole matrix. can be expensive

public:
///	const ElemType* NzValues() const { return Data(); }
///	ElemType* NzValues() { return Data(); }
///
///	size_t NzCount() const
///	{
///		if (GetFormat() == matrixFormatSparseCSC) return GetCompPos()[GetNumCols()] - GetCompPos()[0];
///		if (GetFormat()== matrixFormatSparseCSR) return GetCompPos()[GetNumRows()] - GetCompPos()[0];
///		if (GetFormat() == matrixFormatSparseBSC) return GetBlockSize() * GetNumRows();
///		NOT_IMPLEMENTED;
///	}
///	size_t NzSize() const { return sizeof(ElemType) * NzCount(); }	// actual number of element bytes in use
///
///	size_t GetBlockSize() const { return BaseMatrix<ElemType>::GetBlockSize(); }
///	void SetBlockSize(size_t newBlockSize) { BaseMatrix<ElemType>::SetBlockSize(newBlockSize); }
///
///	size_t* BlockIdsLocation() const
///	{
///		if ((GetFormat() != matrixFormatSparseBSC) && (GetFormat() != matrixFormatSparseBSR))
///			LogicError("CPUSparseMatrix::BlockIdsLocation is only applicable to sparse block formats");
///
///		return GetBlockIds();
///	}
///
///	index_t* MajorIndexLocation() const
///	{
///		return (GetCompId() + 
///			((GetFormat() == matrixFormatSparseCSC || GetFormat() == matrixFormatSparseCSR) ? GetCompPos()[m_sliceOffset] : 0));
///	} // this is the major index, row/col ids in CSC/CSR format
///
///	size_t MajorIndexCount() const { return NzCount(); }
///	size_t MajorIndexSize() const { return MajorIndexCount()*sizeof(index_t); }	// actual number of major index bytes in use
///
///	// Returns the start of the secondary index valid for the slice-view.
///	// Secondary index provides the offset to the data buffer for the values.
///	// E.g. for CSC the first nonzero value of column k is Buffer(SecondIndexLocation[k])
///	index_t* SecondIndexLocation() const { return GetCompPos() + m_sliceOffset; }
///	
///	size_t SecondIndexCount() const
///	{
///		if (GetFormat() & matrixFormatBlock) return NzCount();
///		size_t cnt = (GetFormat() & matrixFormatRowMajor) ? m_numRows : m_numCols;
///		if (cnt > 0) cnt++; // add an extra element on the end for the "max" value
///		return cnt;
///	}
///	// get size for compressed index
///	size_t SecondIndexSize() const { return SecondIndexCount()*sizeof(index_t); }
///
///	size_t RowSize() const { return (GetFormat() & matrixFormatRowMajor) ? SecondIndexSize() : MajorIndexSize(); }
///	size_t ColSize() const { return (GetFormat() & matrixFormatRowMajor) ? MajorIndexSize() : SecondIndexSize(); }
///
///	index_t* RowLocation() const { return (GetFormat() & matrixFormatRowMajor) ? SecondIndexLocation() : MajorIndexLocation(); }
///	index_t* ColLocation() const { return (GetFormat() & matrixFormatRowMajor) ? MajorIndexLocation() : SecondIndexLocation(); }
};

typedef CPUSparseMatrix<float> CPUSingleSparseMatrix;
typedef CPUSparseMatrix<double> CPUDoubleSparseMatrix;

} } }
