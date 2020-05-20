//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include <stdio.h>
#include "CPUMatrix.h"
//#include "GPUMatrix.h"
//#include "GPUSparseMatrix.h"
//#include <map>
//#include <unordered_map>


namespace Microsoft { namespace MSR { namespace CNTK {


template <class ElemType>
class MATH_API CPUSparseMatrix : public BaseMatrix<ElemType>
{
	typedef BaseMatrix<ElemType> Base;

public:
	explicit CPUSparseMatrix(int flags=0) : Base(MatrixFormat(matrixFormatSparse|flags), CPUDEVICE) {}
	CPUSparseMatrix(size_t rows, size_t cols, size_t n=0, int flags=0) : Base(MatrixFormat(matrixFormatSparse|flags), CPUDEVICE) { Create(rows,cols); if (n) Allocate(n); }
///	CPUSparseMatrix(const CPUSparseMatrix<ElemType>& copyFrom) { Assign(copyFrom); }
///	CPUSparseMatrix<ElemType>& operator=(const CPUSparseMatrix<ElemType>& copyFrom) { if (&copyFrom!=this) Assign(copyFrom); return *this; }
///	CPUSparseMatrix(CPUSparseMatrix<ElemType>&& moveFrom) { Assign(moveFrom,true); moveFrom.Release(); }
///	CPUSparseMatrix<ElemType>& operator=(CPUSparseMatrix<ElemType>&& moveFrom) { if (&moveFrom!=this) { Assign(moveFrom); moveFrom.Release(); } return *this; }

public:
///	void SetValue(size_t row, size_t col, ElemType val) { PutItem(row, col, val); }
///	//void SetValue(const CPUMatrix<ElemType>& mat);
///	//void SetValue(const GPUMatrix<ElemType>& mat);
///	void SetValue(const CPUSparseMatrix<ElemType>& mat);
///	//void SetValue(const GPUSparseMatrix<ElemType>& mat);
///
///	void MaskColumnsValue(const CPUMatrix<char>& mask, ElemType val, size_t mcols);
///
///	CPUSparseMatrix<ElemType>& AssignOneHot(const CPUMatrix<ElemType>& a, vector<size_t>& shape, size_t axis);
///	void SetDiagonalValue(ElemType v);
///	void SetDiagonalValue(const CPUMatrix<ElemType>& v);
///
///	CPUSparseMatrix<ElemType>& DoGatherColumnsOf(ElemType beta, const CPUMatrix<ElemType>& idx, const CPUSparseMatrix<ElemType>& a, ElemType alpha);
///	CPUSparseMatrix<ElemType>& DoScatterColumnsOf(ElemType beta, const CPUMatrix<ElemType>& idx, const CPUSparseMatrix<ElemType>& a, ElemType alpha) { NOT_IMPLEMENTED }
///
///	ElemType* Data() const;
///
///	size_t BufferSize() const { return GetSizeAllocated() * sizeof(ElemType); }
///	size_t GetNumElemAllocated() const { return GetSizeAllocated(); }
///
///	CPUSparseMatrix<ElemType> GetColumnSlice(size_t start, size_t len) const;
///	CPUMatrix<ElemType> CopyColumnSliceToDense(size_t start, size_t len) const;
///	void AssignColumnSliceToDense(CPUMatrix<ElemType>& slice, size_t start, size_t len) const;
///
///	CPUMatrix<ElemType> DiagonalToDense() const;
///	CPUMatrix<ElemType> CopyToDense() const { CPUMatrix<ElemType> dm; Base::CopyToDense(dm); return dm; }
///	CPUSparseMatrix<ElemType> CopyToSparse() const { CPUSparseMatrix<ElemType> dm; Base::CopyToSparse(dm); return dm; }
///	CPUSparseMatrix<ElemType> CopyToBlock() const { CPUSparseMatrix<ElemType> dm; Base::CopyToBlock(dm); return dm; }
///	void ConvertToFullBlock() { m_sob->MakeFullBlock(); }
///
///	CPUSparseMatrix<ElemType> Transpose(bool hdr=false) const { CPUSparseMatrix<ElemType> sm; sm.TransposeFrom(*this,hdr); return sm; }
///
///	void SetGaussianRandomValue(ElemType /*mean*/, ElemType /*sigma*/, unsigned long /*seed*/)
///	{
///		NOT_IMPLEMENTED;
///	}
///
///	void SetMatrixFromCSCFormat(const index_t* pCol, const index_t* pRow, const ElemType* pVal, size_t nz, size_t numRows, size_t numCols);	// no slice
///	void SetMatrixFromBSCFormat(const size_t* blockPos, const ElemType* pVal, size_t numBlocks, size_t numRows, size_t numCols);			// no slice
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

	// Allocate actually allocates the storage space for numNzElem elements. This is different than resizing, which changes the dimensions of the underlying matrix.
	// Unfortunately numRows/numCols need to be passed in the case of various matrix formats (e.g., SparseCSC), because some of the dimensions allocated depend on the
	// dimensions of the matrix.
///	void Allocate(size_t numRows, size_t numCols, size_t numNzElem = 10000, bool growOnly = true, bool keepval = false); // matrix format will affect the size to allocate
	// RequireSizeAndAllocate is required by SpasreMatrix since resizing the dimensions and allocating storage are different operations. Since a Resize can entail changing
	// MatrixFormat, we offer an overload for that case.
///	void RequireSizeAndAllocate(size_t numRows, size_t numCols, size_t numNzElem, MatrixFormat mft, bool growOnly = true, bool keepval = true); // matrix format will affect the size to allocate
	// Otherwise we will just use the current MatrixFormat.
	void RequireSizeAndAllocate(size_t numRows, size_t numCols, size_t numNzElem = 10000, bool growOnly = true, bool keepval = false);
	// Sparse matrix RequireSize is similar to dense matrix RequireSize in that it will only allocate the minimum amount of storage required to successfully create the matrix.
	// This is required because some formats (e.g., SparseCSC) require the SecondIndexLocation to have valid data in order to compute m_nz. Otherwise this method would not
	// update the storage at all.
///	void RequireSize(size_t numRows, size_t numCols, size_t numNzElem, MatrixFormat mft, bool growOnly = true);
///	void RequireSize(size_t numRows, size_t numCols, MatrixFormat mft, bool growOnly = true) { return RequireSize(numRows, numCols, 0, mft, growOnly); }
///	void RequireSize(size_t numRows, size_t numCols, bool growOnly = true);
	// Resizes the dimensions of the underlying sparse matrix object. Since the caller may have a hint for m_nz, we allow that to be passed, but this is terrible design. In the
	// future if we want better separation between allocation and resizing, this interface should be updated to be less clunky.
///	void Resize(size_t numRows, size_t numCols, size_t numNz, MatrixFormat mft, bool growOnly = true); // matrix format will affect the size to allocate
///	void Resize(size_t numRows, size_t numCols, size_t numNz, bool growOnly = true);
///	using Base::Resize;

///	void Reset();

///	ElemType operator()(size_t row, size_t col) const { return GetItem(row, col); }

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
