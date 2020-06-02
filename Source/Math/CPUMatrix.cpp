//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "../Common/Basics.h"
#include "../Common/File.h"
#include "CPUMatrix.h"
#include "CPUSparseMatrix.h"
#include "TensorOps.h"
#include <assert.h>
//#include <stdexcept>
//#include <omp.h>
#include <math.h>
//#include <random>
//#include <chrono>
//#include <exception>
//#include <thread>
//#include <iostream>
//#include <algorithm>
#include <numeric>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <cfloat>
#endif

#ifdef LEAKDETECT
#include <vld.h>
#endif

#pragma warning(disable : 4100) // unreferenced formal parameter; "struct TensorOpReduction<ElemType, OPFN, typename ReductionOp, N, -1>" trigger this
#pragma warning(disable : 4127) // conditional expression is constant; "if (sizeof(ElemType)==sizeof(float))" triggers this
#pragma warning(disable : 4244) // unreachable code; triggered for unknown reasons
#pragma warning(disable : 4702) // conversion from 'double' to 'float'

#ifdef USE_MKL
// requires MKLML 0.11 and above
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#include <mkl_service.h>
#else
#ifdef _MSC_VER
// Visual Studio doesn't define standard complex types properly
#define HAVE_LAPACK_CONFIG_H
#define LAPACK_COMPLEX_STRUCTURE
#endif
#include <cblas.h>
//#include <lapacke.h>
#endif

//#define SWAP(a, b)  \
//	{               \
//		(a) ^= (b); \
//		(b) ^= (a); \
//		(a) ^= (b); \
//	}
//#define IDX2C(i, j, ld) (((j) * (ld)) + (i)) // 0 based indexing

namespace Microsoft { namespace MSR { namespace CNTK {

//==============================================================================
//			CPUMatrix<ElemType>
//==============================================================================

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//			Basic Operators
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::GetColumnSlice(size_t start, size_t cols) const
{
	if (start + cols > m_numCols)
		InvalidArgument("The slice (%d+%d) is out of range of the source matrix (%d).", (int)start, (int)cols, (int)m_numCols);

	CPUMatrix<ElemType> slice(*this, true);
	slice.SetColumnSlice(start,cols);
	return slice;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignColumnSlice(const CPUMatrix<ElemType>& fromMatrix, size_t start, size_t len)
{
	CPUMatrix<ElemType> c(fromMatrix, true);
	c.SetSlice(start, len); c.CopyToDense(*this);
	return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::PutColumnSlice(const CPUMatrix<ElemType>& fromMatrix, size_t start, size_t len)
{
	if (len > fromMatrix.GetNumCols())
		InvalidArgument("The slice (%lu) is out of source range %lu)", len, fromMatrix.GetNumCols());
	if (start + len > m_numCols)
		LogicError("The slice (%lu+%lu) is out of destination range (%lu)", start, len, m_numCols);
	if (m_numRows != fromMatrix.m_numRows)
		LogicError("Different number of rows = %lu / %lu", fromMatrix.m_numRows, m_numRows);

	memcpy(GetData() + start*m_numRows, fromMatrix.GetData(), len*m_numRows*sizeof(ElemType));
	return *this;
}

///template <class ElemType>
///void CPUMatrix<ElemType>::CopyColumnsStrided(const CPUMatrix<ElemType>& fromMatrix, size_t numCols, size_t srcNumColsStride, size_t destNumColsStride)
///{
///	if ((((numCols - 1) * srcNumColsStride) + 1) > fromMatrix.m_numCols)
///		LogicError("The numCols to copy and srcNumColsStride specified is out of range of the source matrix");
///	if ((((numCols - 1) * destNumColsStride) + 1) > m_numCols)
///		LogicError("The numCols to copy and srcNumColsStride specified is out of range of the destination matrix");
///	if (m_numRows != fromMatrix.m_numRows)
///		LogicError("The number of rows in source and destination matrices do not match");

///	auto& us = *this;
///	long n = (long) numCols, m = (long) m_numRows;
///#pragma omp parallel for
///	for (long j = 0; j < n; j++)
///	{
///		// four-way unrolling
///		for (size_t i = 0; i < m4; i += 4)
///		{
///			us(i, j*destNumColsStride) = fromMatrix(i, j*srcNumColsStride);
///			us(i+1, j*destNumColsStride) = fromMatrix(i+1, j*srcNumColsStride);
///			us(i+2, j*destNumColsStride) = fromMatrix(i+2, j*srcNumColsStride);
///			us(i+3, j*destNumColsStride) = fromMatrix(i+3, j*srcNumColsStride);
///		}
///		// handle remaining
///		for (size_t i = m4; i < m; i++)
///			us(i, j*destNumColsStride) = fromMatrix(i, j*srcNumColsStride);
///	}
///}

////for each column of a, we add all rows of a to this starting from startIndex
///template <class ElemType>
///CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignToRowSliceValuesOf(const CPUMatrix<ElemType>& a, size_t startIndex, size_t numRows)
///{
///	if (a.GetNumRows() != numRows) LogicError("AddToRowSliceValuesOf: a.GetNumRows() != numRows");
///	if (startIndex + numRows > GetNumRows()) LogicError("AddToRowSliceValuesOf: startIndex + numRows exceeds GetNumRows()");
///	if (a.GetNumCols() != GetNumCols()) LogicError("AddToRowSliceValuesOf: columns does not match");

///	auto& us = *this;
///	long n = (long) a.GetNumCols(), m = (long) numRows;
///#pragma omp parallel for
///	for (long j = 0; j < n; j++)
///	{
///		// four-way unrolling
///		for (size_t i = 0, startRow = startIndex; i < m4; i += 4, startRow += 4)
///		{
///			us(startRow, j) = a(i, j);
///			us(startRow+1, j) = a(i+1, j);
///			us(startRow+2, j) = a(i+2, j);
///			us(startRow+3, j) = a(i+3, j);
///		}
///		// handle remaining stuffs
///		for (size_t i = m4, startRow = startIndex + m4; i < m; i++, startRow++)
///			us(startRow, j) = a(i, j);
///	}
///	return *this;
///}

//for each column of a, we assign numRows starting from startIndex to this
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignRowSliceValuesOf(const CPUMatrix<ElemType>& a, size_t startIndex, size_t numRows)
{
	if (startIndex + numRows > a.GetNumRows())
		LogicError("AssignRowSliceValuesOf: startIndex + numRows exceeds a.GetNumRows()");

	Resize(numRows, a.GetNumCols());

	long n = (long) a.GetNumCols(); // note: OpenMP requires loop indices to be long, not size_t
	long k = (long) a.GetNumRows();

#pragma omp parallel for
	for (long j = 0; j < n; j++)
	{
		// memory copy might be faster?
		memcpy(GetData() + j*numRows, a.GetData() + j*k + startIndex, sizeof(ElemType) * numRows);

		// //four-way unrolling
		// for (long i=0, startRow = startIndex; i<m4; i+=4, startRow+=4)
		// {
		//    us(i,j) = a(startRow,j);
		//    us(i+1,j) = a(startRow+1,j);
		//    us(i+2,j) = a(startRow+2,j);
		//    us(i+3,j) = a(startRow+3,j);
		// }
		// //handle remaining stuffs
		// for (long i=m & ~3, startRow = startIndex+m4; i<m; i++, startRow++)
		// {
		//    us(i,j) = a(startRow,j);
		// }
	}

	return *this;
}

//for the row slice of this starting from startIndex we add a to it.
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AddToRowSliceValuesOf(const CPUMatrix<ElemType>& a, size_t startIndex, size_t numRows)
{
	if (a.IsEmpty())
		LogicError("AddToRowSliceValuesOf: input matrix a is empty");

	if (a.GetNumRows() != numRows)
		LogicError("AddToRowSliceValuesOf: a.GetNumRows() != numRows");

	if (startIndex + numRows > GetNumRows())
		LogicError("AddToRowSliceValuesOf: startIndex + numRows exceeds GetNumRows()");

	if (a.GetNumCols() != GetNumCols())
		LogicError("AddToRowSliceValuesOf: columns does not match");

	long n = (long) a.GetNumCols(), m = (long) numRows;
	long m4 = m & ~3;

	auto& us = *this;

#pragma omp parallel for
	for (long j = 0; j < n; j++)
	{
		// four-way unrolling
		for (long i = 0, startRow = (long) startIndex; i < m4; i += 4, startRow += 4)
		{
			us(startRow, j) += a(i, j);
			us(startRow+1, j) += a(i+1, j);
			us(startRow+2, j) += a(i+2, j);
			us(startRow+3, j) += a(i+3, j);
		}
		// handle remaining stuffs
		for (long i = m4, startRow = (long) startIndex + m4; i < m; i++, startRow++)
		{
			us(startRow, j) += a(i, j);
		}
	}

	return *this;
}

//for each column of this, we add row slice of a starting from startIndex
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AddWithRowSliceValuesOf(const CPUMatrix<ElemType>& a, size_t startIndex, size_t numRows)
{
	if (a.IsEmpty())
		LogicError("AddWithRowSliceValuesOf: input matrix a is empty");

	if (GetNumRows() != numRows)
		LogicError("AddWithRowSliceValuesOf: GetNumRows() != numRows");

	if (startIndex + numRows > a.GetNumRows())
		LogicError("AddWithRowSliceValuesOf: startIndex + numRows exceeds a.GetNumRows()");

	if (a.GetNumCols() != GetNumCols())
		LogicError("AddWithRowSliceValuesOf: columns does not match");

	long n = (long) a.GetNumCols(), m = (long) numRows;
	long m4 = m & ~3;

	auto& us = *this;

#pragma omp parallel for
	for (long j = 0; j < n; j++)
	{
		// four-way unrolling
		for (long i = 0, startRow = (long) startIndex; i < m4; i += 4, startRow += 4)
		{
			us(i, j) += a(startRow, j);
			us(i+1, j) += a(startRow+1, j);
			us(i+2, j) += a(startRow+2, j);
			us(i+3, j) += a(startRow+3, j);
		}
		// handle remaining stuffs
		for (long i = m4, startRow = (long) startIndex + m4; i < m; i++, startRow++)
		{
			us(i, j) += a(startRow, j);
		}
	}
	return *this;
}

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::Diagonal() const
{
	size_t n = GetDiagSize();
	CPUMatrix<ElemType> diag(1,n);
	const ElemType* pi = GetData();
	ElemType* po = diag.GetData();
	for (size_t j=0; j<n; ++j) { *po++ = *pi; pi += m_numRows + 1; }
	return diag;
}

///template <class ElemType>
///void CPUMatrix<ElemType>::MinusOneAt(CPUMatrix<ElemType>& c, size_t position)
///{
///	if (position < c.GetNumElements())
///		c.GetData()[position] -= 1.0;
///	else
///		RuntimeError("MinusOneAt: position is out of CPU matrix size");
///}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignRepeatOf(const CPUMatrix<ElemType>& a, size_t numRowRepeats, size_t numColRepeats)
{
	if (this == &a)
		LogicError("AssignRepeatOf: a is the same as [this]. Does not support inplace repeat");

	if (a.IsEmpty())
		LogicError("AssignRepeatOf; Matrix a is empty");

	Resize(a.GetNumRows()*numRowRepeats, a.GetNumCols()*numColRepeats);
	long n = (long) a.GetNumCols(), m = (long) a.GetNumRows();
	long m4 = m & ~3;
	auto& us = *this;

#pragma omp parallel for
	for (long q = 0; q < numColRepeats; q++)
	{
		for (long p = 0; p < numRowRepeats; p++)
		{
			long colOffset = q * n;

			for (long j = 0; j < n; j++, colOffset++)
			{
				long rowOffset = p * m;

				// four-way unrolling
				for (long i = 0; i < m4; i += 4, rowOffset += 4)
				{
					us(rowOffset, colOffset) = a(i, j);
					us(rowOffset+1, colOffset) = a(i+1, j);
					us(rowOffset+2, colOffset) = a(i+2, j);
					us(rowOffset+3, colOffset) = a(i+3, j);
				}
				// handle remaining stuffs
				for (long i = m4; i < m; i++, rowOffset++)
				{
					us(rowOffset, colOffset) = a(i, j);
				}
			}
		}
	}

	return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AddToRowRepeatValuesOf(const CPUMatrix<ElemType>& a, size_t numRepeats)
{
	if (a.IsEmpty())
		LogicError("AddToRowRepeatValuesOf: input matrix a is empty");

	if (a.GetNumRows() != GetNumRows() * numRepeats)
		LogicError("AddToRowRepeatValuesOf: a.GetNumRows() != GetNumRows() * numRepeats");

	auto& us = *this;
	long n = (long) a.GetNumCols(), m = (long) GetNumRows();
	long m4 = m & ~3;
#pragma omp parallel for
	for (long j = 0; j < n; j++)
	{
		// four-way unrolling
		for (long i = 0; i < m4; i += 4)
		{
			for (long k = 0; k < numRepeats; k++)
			{
				us(i, j) += a(k*m + i, j);
				us(i+1, j) += a(k*m + i+1, j);
				us(i+2, j) += a(k*m + i+2, j);
				us(i+3, j) += a(k*m + i+3, j);
			}
		}
		// handle remaining stuffs
		for (long i = m4; i < m; i++)
			for (long k = 0; k < numRepeats; k++)
				us(i, j) += a(k*m + i, j);
	}
	return *this;
}

///template <class ElemType>
///CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignPositiveAndShiftedNegSample(const CPUMatrix<ElemType>& a, size_t posNumber, size_t negNumber, size_t shiftNumber)
///{
///	a;
///	posNumber;
///	negNumber;
///	shiftNumber;
///	NOT_IMPLEMENTED;
///}

///template <class ElemType>
///CPUMatrix<ElemType>& CPUMatrix<ElemType>::AddFoldedPositiveAndShiftedNegSample(const CPUMatrix<ElemType>& a, size_t posNumber, size_t negNumber, size_t shiftNumber)
///{
///	a;
///	posNumber;
///	negNumber;
///	shiftNumber;
///	NOT_IMPLEMENTED;
///}

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::Transpose()
{
	CPUMatrix<ElemType> c;
	if (!IsEmpty()) TransposeTo(c);
	return c;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignTransposeOf(const CPUMatrix<ElemType>& a)
{
	if (&a != this) a.TransposeTo(*this);
	else Assign(TransposeTo(CPUMatrix<ElemType>()),true);
	return *this;
}

//// dst[i] = src[i] * alpha + dst[i] * beta
//// scale a column vector and add it to another
//// The usual special case: If beta = 0, then dst[] is not read, and may be uninitialized or NaN.
///template <class ElemType>
///static void ScaleAndAddColumn(ElemType beta, ElemType* dst, const ElemType* src, size_t numRows, ElemType alpha)
///{
///	if (alpha != 1) // rare case: just do the full thing
///		for (size_t i = 0; i < numRows; i++) dst[i] = beta * dst[i] + alpha * src[i];
///	else if (beta == 1) // used in backprop
///		for (size_t i = 0; i < numRows; i++) dst[i] += src[i];
///	else if (beta == 0) // plain assignment
///		memcpy(dst, src, sizeof(ElemType) * numRows);
///	else // alpha=1, arbitrary beta: also rare case
///		for (size_t i = 0; i < numRows; i++) dst[i] = beta * dst[i] + src[i];
///}

//// *this[:,j] = a[:,idx[j]] * alpha + *this[:,j] * beta
///template <class ElemType>
///CPUMatrix<ElemType>& CPUMatrix<ElemType>::DoGatherColumnsOf(ElemType beta, const CPUMatrix<ElemType>& idx, const CPUMatrix<ElemType>& a, ElemType alpha)
///{
///	if (idx.GetNumRows() != 1) // index is 1-dimensional only
///		InvalidArgument("DoGatherColumnsOf: Map must be a row vector");

///	if (beta) VerifySize(a.GetNumRows(), idx.GetNumCols());
///	else Resize(a.GetNumRows(), idx.GetNumCols());

///	auto& us = *this;
///	// race-condition consideration: Since this loops over independent output columns, this has no race condition. Cf. DoScatterColumnsOf().
///#pragma omp parallel for // TODO: Depending in circumstance, it may be more efficient to parallelize over rows.
///	foreach_column(jOut, us)
///	{
///		auto jInF = idx(0, jOut);						// this is the column we need to get
///		if (std::isnan(jInF) || jInF < 0) continue;		// negative index means gap
///		size_t jIn = (size_t)jInF;
///		if (jIn >= a.GetNumCols())
///			InvalidArgument("DoGatherColumnsOf: Map out of bounds. %ld >= %ld", (long int)jIn, (long int)a.GetNumCols());
///		ScaleAndAddColumn(beta, &us(0,jOut), &a(0,jIn), us.GetNumRows(), alpha);
///	}
///	return *this;
///}

//// *this[:,idx[j]] = a[:,j] * alpha + *this[:,idx[j]] * beta
///template <class ElemType>
///CPUMatrix<ElemType>& CPUMatrix<ElemType>::DoScatterColumnsOf(ElemType beta, const CPUMatrix<ElemType>& idx, const CPUMatrix<ElemType>& a, ElemType alpha)
///{
///	if (idx.GetNumRows() != 1) // index is 1-dimensional only
///		InvalidArgument("DoScatterColumnsOf: Map must be a row vector");
///	if (idx.GetNumCols() != a.GetNumCols())
///		InvalidArgument("DoScatterColumnsOf: Map must have width of input vector");
///	if (a.GetNumRows() != GetNumRows())
///		InvalidArgument("DoScatterColumnsOf: Output must have same height as input vector");

///	auto& us = *this;

///	// pre-scale with beta upfront
///	// Scatter may add more than one source column to the same target, so we must pre-scale with beta, and then just keep adding.
///	Scale(beta, us); // if beta is 0, then this will be a memset()

///	ScatterValues(idx.GetData(), a.GetData(), us.GetData(), alpha, idx.GetNumCols(), a.GetNumRows(), GetNumCols(), idx.GetNumRows());

///	return *this;
///}

template <class ElemType>
void CPUMatrix<ElemType>::SetValue(ElemType v)
{
	if (IsEmpty()) LogicError("SetValue; Matrix is empty");

	bool isFinite = std::numeric_limits<ElemType>::is_integer || std::isfinite((double) v);
	if (isFinite && v==0) { memset(GetData(), 0, sizeof(ElemType) * GetNumElements()); return; }

	ElemType* ptr = GetData();
	long m = (long) GetNumElements();
	long m4 = m & ~3;

	// 2-way thread parallelism is sufficient for the memory bound
	// operation of just setting the values of an array.
	const unsigned SETVALUE_NUM_THREADS = 2;
	//UNUSED(SETVALUE_NUM_THREADS); // in case OMP is turned off.
#pragma omp parallel for num_threads(SETVALUE_NUM_THREADS)
	// four-way unrolling
	for (long i = 0; i < m4; i += 4)
	{
		ptr[i] = v;
		ptr[i+1] = v;
		ptr[i+2] = v;
		ptr[i+3] = v;
	}
	// handle remaining stuffs
	for (long i = m4; i < m; i++) ptr[i] = v;
}

///template <class ElemType>
///void CPUMatrix<ElemType>::MaskColumnsValue(const CPUMatrix<char>& columnsMask, ElemType val, size_t numColsPerMaskEntry)
///{
///	if (GetNumCols() != (columnsMask.GetNumCols() * numColsPerMaskEntry))
///		RuntimeError("MaskColumnsValue; Matrix number of columns must equal 'column mask number of columns * numColsPerMaskEntry'");

///	auto& us = *this;
///	long n = (long) columnsMask.GetNumCols(), m = (long) GetNumRows();
///#pragma omp parallel for
///	for (long j = 0; j < n; j++)
///	{
///		if (columnsMask(0, j) == 1) continue;
///		for (long k = 0; k < numColsPerMaskEntry; ++k)
///		{
///			for (size_t i = 0; i < m4; i += 4)
///			{
///				us(i, (j*numColsPerMaskEntry) + k) = val;
///				us(i+1, (j*numColsPerMaskEntry) + k) = val;
///				us(i+2, (j*numColsPerMaskEntry) + k) = val;
///				us(i+3, (j*numColsPerMaskEntry) + k) = val;
///			}
///			for (size_t i = m4; i < m; i++)
///				us(i, (j*numColsPerMaskEntry) + k) = val;
///		}
///	}
///}

template <class ElemType>
void CPUMatrix<ElemType>::SetColumn(size_t col, ElemType val)
{
	if (IsEmpty()) return;
		//LogicError("SetColumn; Matrix is empty");

	auto& us = *this;
	long m = (long) GetNumRows();
	long m4 = m & ~3;
#pragma omp parallel for
	// four-way unrolling
	for (long i = 0; i < m4; i += 4)
	{
		us(i, col) = val;
		us(i+1, col) = val;
		us(i+2, col) = val;
		us(i+3, col) = val;
	}
	// handle remaining stuffs
	for (long i = m4; i < m; i++) us(i, col) = val;
}

template <class ElemType>
void CPUMatrix<ElemType>::SetColumn(size_t col, const ElemType* p)
{
	if (IsEmpty()) return;
		//LogicError("SetColumn; Matrix is empty");
	if (p == NULL) return;

	auto& us = *this;
	long m = (long) GetNumRows();
	long m4 = m & ~3;
#pragma omp parallel for
	for (long i = 0; i < m4; i += 4)
	{
		us(i, col) = p[i];
		us(i+1, col) = p[i+1];
		us(i+2, col) = p[i+2];
		us(i+3, col) = p[i+3];
	}
	for (long i = m4; i < m; i++)
		us(i, col) = p[i];
}

template <class ElemType>
void CPUMatrix<ElemType>::SetColumn(size_t col, const CPUMatrix<ElemType>& mat)
{
	if (IsEmpty()) return;
		//LogicError("SetColumn; Matrix is empty");
	if (mat.GetNumRows() != GetNumRows() || mat.GetNumCols() != 1)
		LogicError("SetColumn; The matrix has incorrect number of rows or columns");

	auto& us = *this;
	long m = (long) GetNumRows();
	long m4 = m & ~3;
#pragma omp parallel for
	// four-way unrolling
	for (long i = 0; i < m4; i += 4)
	{
		us(i, col) = mat(i, 0);
		us(i+1, col) = mat(i+1, 0);
		us(i+2, col) = mat(i+2, 0);
		us(i+3, col) = mat(i+3, 0);
	}
	// handle remaining stuffs
	for (long i = m4; i < m; i++)
	{
		us(i, col) = mat(i, 0);
	}
}

template <class ElemType>
void CPUMatrix<ElemType>::SetValue(const CPUMatrix<ElemType>& mat)
{
	if (&mat != this) SetValue(mat.GetNumRows(), mat.GetNumCols(), mat.GetData(), mat.GetFormat());
}

///#if 0
///template <class ElemType>
///void CPUMatrix<ElemType>::SetValue(const GPUMatrix<ElemType>& /*copyFrom*/)
///{
///	NOT_IMPLEMENTED;
///}

///template <class ElemType>
///void CPUMatrix<ElemType>::SetValue(const CPUSparseMatrix<ElemType>& copyFrom)
///{
///	copyFrom.AssignColumnSliceToDense(*this, 0, copyFrom.GetNumCols());
///}

///template <class ElemType>
///void CPUMatrix<ElemType>::SetValue(const GPUSparseMatrix<ElemType>& /*copyFrom*/)
///{
///	NOT_IMPLEMENTED;
///}
///#endif

template <class ElemType>
void CPUMatrix<ElemType>::SetValue(size_t rows, size_t cols, ElemType* p, int flags)
{
	if (p==nullptr && rows*cols > 0)
		InvalidArgument("SetValue; Reference data is null");

	Init(matrixFormatDenseCol, CPUDEVICE);
	if (flags & matrixFlagExternalBuffer) { Assign(rows, cols, p, flags); return; }
	Resize(rows, cols); if (IsEmpty()) return;
	if ((flags & matrixFormatRowMajor)==0) { memcpy(GetData(), p, rows*cols*sizeof(ElemType)); return; }

	// transpose
	auto& us = *this;
	ElemType* ptr = GetData();
	if (std::is_same<ElemType, double>::value)
	{
#pragma omp parallel for
		foreach_column (j, us)
		{
			cblas_dcopy((int)rows, reinterpret_cast<double*>(p + j), (int)cols, reinterpret_cast<double*>(ptr + ColumnPos(j)), 1);
		}
	}
	else if (std::is_same<ElemType, float>::value)
	{
#pragma omp parallel for
		foreach_column (j, us)
		{
			cblas_scopy((int)rows, reinterpret_cast<float*>(p + j), (int)cols, reinterpret_cast<float*>(ptr + ColumnPos(j)), 1);
		}
	}
	else RuntimeError("SetValue; Unsupported data type");
}

template <class ElemType>
void CPUMatrix<ElemType>::SetDiagonalValue(ElemType v)
{
	auto& us = *this;
	long m = static_cast<long>(GetDiagSize());
	long m4 = m & ~3;
#pragma omp parallel for
	// four-way unrolling
	for (long i = 0; i < m4; i += 4)
	{
		us(i, i) = v;
		us(i+1, i+1) = v;
		us(i+2, i+2) = v;
		us(i+3, i+3) = v;
	}
	// handle remaining stuffs
	for (long i = m4; i < m; i++)
	{
		us(i, i) = v;
	}
}

template <class ElemType>
void CPUMatrix<ElemType>::SetDiagonalValue(const CPUMatrix<ElemType>& vector)
{
	if (IsEmpty() || vector.IsEmpty())
		LogicError("SetDiagonalValue; Matrix is empty");

	if (vector.GetNumRows() != 1 && vector.GetNumCols() != 1)
		LogicError("SetDiagonalValue: input vector must be a vector");

	if (vector.GetNumElements() == 1) // reduce to simple form
		SetDiagonalValue(vector(0, 0));
	else if (vector.GetNumRows() != GetDiagSize() && vector.GetNumCols() != GetDiagSize())
		LogicError("SetDiagonalValue: input vector's dimension does not agree with [this]");
	else
	{
		auto& us = *this;

		long m = (long) GetDiagSize();
		long m4 = m & ~3;
		if (vector.GetNumRows() == 1) // row vector
		{
#pragma omp parallel for
			// four-way unrolling
			for (long i = 0; i < m4; i += 4)
			{
				us(i, i) = vector(0, i);
				us(i+1, i+1) = vector(0, i+1);
				us(i+2, i+2) = vector(0, i+2);
				us(i+3, i+3) = vector(0, i+3);
			}
			// handle remaining stuffs
			for (long i = m4; i < m; i++)
			{
				us(i, i) = vector(0, i);
			}
		}
		else
		{
#pragma omp parallel for
			// four-way unrolling
			for (long i = 0; i < m4; i += 4)
			{
				us(i, i) = vector(i, 0);
				us(i+1, i+1) = vector(i+1, 0);
				us(i+2, i+2) = vector(i+2, 0);
				us(i+3, i+3) = vector(i+3, 0);
			}
			// handle remaining stuffs
			for (long i = m4; i < m; i++)
			{
				us(i, i) = vector(i, 0);
			}
		}
	}
}

template <class ElemType>
void CPUMatrix<ElemType>::SetUniformRandomValue(ElemType low, ElemType high, unsigned long seed)
{
	if (IsEmpty())
		LogicError("SetUniformRandomValue; Matrix is empty");

	std::mt19937_64 generator;
	generator.seed(seed == USE_TIME_BASED_SEED ? (unsigned long) time(NULL) : seed);
	std::uniform_real_distribution<double> r((double)low, (double)high);

	ElemType* bufPtr = GetData();
	long m = (long) GetNumElements();
	long m4 = m & ~3;
	// four-way unrolling
	for (long i = 0; i < m4; i += 4)
	{
		bufPtr[i]     = (ElemType)r(generator);
		bufPtr[i+1] = (ElemType)r(generator);
		bufPtr[i+2] = (ElemType)r(generator);
		bufPtr[i+3] = (ElemType)r(generator);
	}
	// handle remaining stuffs
	for (long i = m4; i < m; i++)
	{
		bufPtr[i] = (ElemType)r(generator);
	}
}


template <class ElemType>
void CPUMatrix<ElemType>::SetUniformRandomValue(RNGHandle& rngHandle, ElemType low, ElemType high)
{
	if (IsEmpty())
		LogicError("SetUniformRandomValue; Matrix is empty");

	CPURNGHandle* cpuRNGHandle = dynamic_cast<CPURNGHandle*>(&rngHandle);
	if (cpuRNGHandle == nullptr)
		LogicError("rngHandle must be a CPURNGHandle");

	std::uniform_real_distribution<double> r((double)low, (double)high);
	std::generate(GetData(), GetData() + GetNumElements(), [&cpuRNGHandle, &r]() {return (ElemType)r(cpuRNGHandle->Generator()); });
}

template <class ElemType>
void CPUMatrix<ElemType>::SetGaussianRandomValue(RNGHandle& rngHandle, ElemType mean, ElemType stdev)
{
	if (IsEmpty())
		LogicError("SetGaussianRandomValue; Matrix is empty");

	CPURNGHandle* cpuRNGHandle = dynamic_cast<CPURNGHandle*>(&rngHandle);
	if (cpuRNGHandle == nullptr)
		LogicError("rngHandle must be a CPURNGHandle");

	std::normal_distribution<double> r((double)mean, (double)stdev);
	auto n = SizeMultipleOf(GetNumElements(), 2);
	std::generate(GetData(), GetData() + n, [&cpuRNGHandle, &r]() {return (ElemType)r(cpuRNGHandle->Generator()); });
}

template <class ElemType>
void CPUMatrix<ElemType>::SetGumbelRandomValue(RNGHandle& rngHandle, ElemType loc, ElemType scale)
{
	if (IsEmpty())
		LogicError("SetGumbelRandomValue; Matrix is empty");

	CPURNGHandle* cpuRNGHandle = dynamic_cast<CPURNGHandle*>(&rngHandle);
	if (cpuRNGHandle == nullptr)
		LogicError("rngHandle must be a CPURNGHandle");

	std::uniform_real_distribution<double> r(0, 1);
	std::generate(GetData(), GetData() + GetNumElements(), [&cpuRNGHandle, &r, loc, scale]() {return (ElemType)(loc - scale * log(-log1p(-r(cpuRNGHandle->Generator())))); });
}


template <class ElemType>
void CPUMatrix<ElemType>::SetGaussianRandomValue(ElemType mean, ElemType sigma, unsigned long seed)
{
	if (sigma <= 0)
		InvalidArgument("SetGaussianRandomValue: sigma must be a positive value");

	if (IsEmpty())
		LogicError("SetGaussianRandomValue; Matrix is empty");

	auto& us = *this;

	std::mt19937_64 generator(seed == USE_TIME_BASED_SEED ? (unsigned long) time(NULL) : seed);
	std::normal_distribution<double> r((double)mean, (double)sigma);

	// #pragma omp parallel for is not thread safe. Also the results would not be deterministic
	foreach_coord (i, j, us)
	{
		us(i, j) = (ElemType)r(generator);
	}
}

template <class ElemType>
void CPUMatrix<ElemType>::SetTruncatedNormalRandomValue(ElemType mean, ElemType sigma, unsigned long seed)
{
	if (sigma <= 0)
		InvalidArgument("SetTruncatedNormalRandomValue: sigma must be a positive value");

	if (IsEmpty())
		LogicError("SetTruncatedNormalRandomValue; Matrix is empty");

	auto& us = *this;

	std::mt19937_64 generator(seed == USE_TIME_BASED_SEED ? (unsigned long)time(NULL) : seed);
	std::normal_distribution<double> r((double)mean, (double)sigma);

	const ElemType high = mean + 2 * sigma;
	const ElemType low = mean - 2 * sigma;
	// #pragma omp parallel for is not thread safe. Also the results would not be deterministic
	foreach_coord(i, j, us)
	{
		ElemType tmp = 0;
		do { tmp = (ElemType)r(generator); }
		while (tmp < low || tmp > high ); // Rejection sampling is fine here because the acceptance probability is about 0.9545
		us(i, j) = tmp;
	}
}

template <class ElemType>
void CPUMatrix<ElemType>::AddGaussianRandomValue(ElemType mean, ElemType sigma, unsigned long seed)
{
	if (sigma <= 0)
		InvalidArgument("SetUniformRandomValue: sigma must be a positive value");

	if (IsEmpty())
		LogicError("SetUniformRandomValue; Matrix is empty");

	auto& us = *this;
	std::mt19937_64 generator;
	generator.seed(seed == USE_TIME_BASED_SEED ? (unsigned long) time(NULL) : seed);
	std::normal_distribution<double> r((double)mean, (double)sigma);

	long m = (long) GetNumRows(), n = (long) GetNumCols();
	long m4 = m & ~3;
	for (long j = 0; j < n; j++)
	{
		// four-way unrolling
		for (long i = 0; i < m4; i += 4)
		{
			us(i, j)     = (ElemType)r(generator);
			us(i+1, j) = (ElemType)r(generator);
			us(i+2, j) = (ElemType)r(generator);
			us(i+3, j) = (ElemType)r(generator);
		}
		// handle remaining stuffs
		for (long i = m4; i < m; i++) us(i, j) = r(generator);
	}
}

//maskRate: percentage of values masked out (similar to dropout rate)
//scaleValue: which scale value to set to the left ones (unmasked items).
template <class ElemType>
void CPUMatrix<ElemType>::SetUniformRandomMask(ElemType maskRate, ElemType scaleValue, RNGHandle& rngHandle)
{
	if (IsEmpty())
		LogicError("SetUniformRandomValue; Matrix is empty");

	CPURNGHandle* cpuRNGHandle = dynamic_cast<CPURNGHandle*>(&rngHandle);
	if (cpuRNGHandle == nullptr)
		LogicError("rngHandle must be a CPURNGHandle");

	auto& us = *this;
	std::uniform_real_distribution<double> r(0, 1);
	long m = (long) GetNumRows(), n = (long) GetNumCols();
	long m4 = m & ~3;
	ElemType v;
	for (long j = 0; j < n; j++)
	{
		// four-way unrolling
		for (long i = 0; i < m4; i += 4)
		{
			v = (ElemType)r(cpuRNGHandle->Generator());
			us(i, j) = v <= maskRate ? (ElemType)0 : scaleValue;
			v = r(cpuRNGHandle->Generator());
			us(i+1, j) = v <= maskRate ? (ElemType)0 : scaleValue;
			v = r(cpuRNGHandle->Generator());
			us(i+2, j) = v <= maskRate ? (ElemType)0 : scaleValue;
			v = r(cpuRNGHandle->Generator());
			us(i+3, j) = v <= maskRate ? (ElemType)0 : scaleValue;
		}
		// handle remaining stuffs
		for (long i = m4; i < m; i++)
		{
			v = (ElemType)r(cpuRNGHandle->Generator());
			us(i, j) = v <= maskRate ? (ElemType)0 : scaleValue;
		}
	}
}

template <class ElemType>
ElemType CPUMatrix<ElemType>::Adagrad(CPUMatrix<ElemType>& gradients, bool needAveMultiplier)
{
	ElemType aveMultiplier = 0;

	if (IsEmpty() || gradients.GetNumCols() != GetNumCols() || gradients.GetNumRows() != GetNumRows())
	{
		Reset();
		Resize(gradients.GetNumRows(), gradients.GetNumCols());
	}
	if (GetNumRows() != gradients.GetNumRows() || GetNumCols() != gradients.GetNumCols())
		LogicError("The matrix gradients must have the same rows and columns as this matrix");

	ElemType *a = GetData(), *d_v = gradients.GetData();
	size_t n = GetNumElements();

	const ElemType floor = 1e-16f;
	ElemType a0, a1, a2, a3;

	// disable omp here because aveMultiper needs to be added atomically. however, it seems the result is incorrect even if rmp atomic and amp critical are used.
	// #pragma omp parallel for
	for (long i = 0; i < (n & ~3); i += 4) // four-way unrolling
	{
		a[i] += d_v[i] * d_v[i];
		a[i+1] += d_v[i+1] * d_v[i+1];
		a[i+2] += d_v[i+2] * d_v[i+2];
		a[i+3] += d_v[i+3] * d_v[i+3];

		a0 = sqrt(a[i] + floor);
		a1 = sqrt(a[i+1] + floor);
		a2 = sqrt(a[i+2] + floor);
		a3 = sqrt(a[i+3] + floor);

		d_v[i] /= a0;
		d_v[i+1] /= a1;
		d_v[i+2] /= a2;
		d_v[i+3] /= a3;

		if (needAveMultiplier)
			aveMultiplier += 1/a0 + 1/a1 + 1/a2 + 1/a3;
	}
	// get the last few elements if any
	for (long i = n & ~3; i < n; i++)
	{
		a[i] += d_v[i] * d_v[i];
		a0 = sqrt(a[i] + floor);
		d_v[i] /= a0;
		if (needAveMultiplier) aveMultiplier += 1/a0;
	}
	if (needAveMultiplier && n > 0) return aveMultiplier/n;
	return 1;
}

template <class ElemType>
void CPUMatrix<ElemType>::FSAdagrad(CPUMatrix<ElemType>& gradients,
									CPUMatrix<ElemType>& functionValues,
									ElemType learnRatePerSample,
									ElemType momentum,
									ElemType adaWeight,
									ElemType adaMul,
									ElemType unitGainFactor)
{
	size_t numColsNeeded = 2 * gradients.GetNumCols();

	if (IsEmpty() || (GetNumCols() < numColsNeeded))
	{
		Reset();
		Resize(gradients.GetNumRows(), numColsNeeded);
	}
	if (GetNumRows() != gradients.GetNumRows() || GetNumCols() != numColsNeeded)
		LogicError("The matrix gradients does not have expected dimensions");

	size_t n = gradients.GetNumElements();
	ElemType* grad = gradients.GetData();
	ElemType* smoothAda = GetData();
	ElemType* smoothMom = GetData() + n;
	ElemType* val = functionValues.GetData();

	// TODO: Unroll 4-times for better performance leveraging vectorization
#pragma omp parallel for
	for (long i = 0; i < n; i++)
	{
		ElemType g = grad[i];
		ElemType adaSqr = adaWeight * smoothAda[i] + (1.0f - adaWeight) * g * g;
		smoothAda[i] = adaSqr;
		if (adaSqr != 0.0f)
		{
			ElemType ada = sqrt(adaSqr);
			ElemType w = adaMul * ((ElemType) 1.0 / ada);
			if (w > 10.0f) w = 10.0f;
			g *= w;
		}
		if (momentum > 0.0f)
		{
			g = momentum * smoothMom[i] + unitGainFactor * g;
			smoothMom[i] = g;
		}
		g *= learnRatePerSample;
		val[i] -= g;
	}
}

template <class ElemType>
void CPUMatrix<ElemType>::Adam(CPUMatrix<ElemType>& gradients, CPUMatrix<ElemType>& functionValues, ElemType learnRatePerSample,
	ElemType momentum, ElemType adaWeight, ElemType adaMul, ElemType epsilon, ElemType unitGainFactor, bool adamax)
{
	size_t numColsNeeded = 2 * gradients.GetNumCols();
	if (IsEmpty() || (GetNumCols() < numColsNeeded))
	{
		Reset();
		Resize(gradients.GetNumRows(), numColsNeeded);
	}
	if (GetNumRows() != gradients.GetNumRows() || GetNumCols() != numColsNeeded)
		LogicError("The matrix gradients does not have expected dimensions");

	size_t n = gradients.GetNumElements();
	ElemType* grad = gradients.GetData();
	ElemType* smoothAda = GetData();
	ElemType* smoothMom = GetData() + n;
	ElemType* val = functionValues.GetData();

	// TODO: Unroll 4-times for better performance leveraging vectorization
#pragma omp parallel for
	for (long i = 0; i < n; i++)
	{
		ElemType g = grad[i];
		ElemType ada;
		if (!adamax)
		{
			ElemType adaSqr = adaWeight * smoothAda[i] + (1.0f - adaWeight) * g * g;
			smoothAda[i] = adaSqr;
			ada = sqrt(adaSqr);
		}
		else
			ada = smoothAda[i] = std::max(adaWeight * smoothAda[i], fabs_(g));

		ElemType w = adaMul * (ElemType)( 1.0 / (ada + epsilon));
		g = momentum * smoothMom[i] + unitGainFactor * g;
		smoothMom[i] = g;
		val[i] -= g * w * learnRatePerSample;
	}
}

template <class ElemType>
ElemType CPUMatrix<ElemType>::RmsProp(CPUMatrix<ElemType>& gradients,
										ElemType RMS_GAMMA,
										ElemType RMS_WGT_INC,
										ElemType RMS_WGT_MAX,
										ElemType RMS_WGT_DEC,
										ElemType RMS_WGT_MIN,
										bool needAveMultiplier,
										bool initialized)
{
	const ElemType floor = 1e-6f;

	size_t n = gradients.GetNumElements();
	ElemType* curr_grad = gradients.GetData();

	if (IsEmpty() || GetNumCols() < gradients.GetNumCols() * 3 || !initialized)
	{
		Reset();
		Resize(gradients.GetNumRows(), gradients.GetNumCols() * 3);

		ElemType* avars = GetData();         // accumulated variances for RMS scaling
		ElemType* steps = GetData() + 2 * n; // current step size

		// initialize moving average of gradient-squared
		for (long i = 0; i < n; i++)
			avars[i] = curr_grad[i] * curr_grad[i];

		// initialize starting step size
		for (long i = 0; i < n; i++) steps[i] = ElemType(0.02);
	}

	ElemType* avars = GetData();         // accumulated variances for RMS scaling
	ElemType* signs = GetData() + n;     // sign of previous gradient
	ElemType* steps = GetData() + 2 * n; // current step size

	if (GetNumRows() != gradients.GetNumRows() || GetNumCols() != gradients.GetNumCols() * 3)
		LogicError("The matrix gradients does not have expected dimensions");

	ElemType ONE_MINUS_GAMMA = ElemType(1.0) - RMS_GAMMA;
	// int upd[] = {
	//    2,2,0,
	//    2,2,0,
	//    1,1,1,
	//    2,2,0,
	//    1,2,1,
	//    0,2,2,
	//    1,1,1,
	//    0,2,2,
	//    0,2,2,
	// };

	//      for (long i=0; i<n; i++)
	//      {
	//          avars[i] = RMS_GAMMA * avars[i] + ONE_MINUS_GAMMA * (curr_grad[i] * curr_grad[i]);
	//    // grad sign base 3: 0->neg, 1->zero, 2->pos
	//    const int grad_sign = 1 + (ElemType(0) < curr_grad[i]) - (curr_grad[i] < ElemType(0));

	//    // signs[i] contains three consecutive grad_sign
	//    signs[i]  = 3*(int(signs[i]) % 9) + grad_sign;

	//    switch(upd[int(signs[i])])
	//    {
	//    case 0:
	//        steps[i] = max(steps[i] * RMS_WGT_DEC, RMS_WGT_MIN);
	//        break;
	//    case 2:
	//        steps[i] = min(steps[i] * RMS_WGT_INC, RMS_WGT_MAX);
	//        break;
	//    }
	//    curr_grad[i] *= steps[i] / sqrt(avars[i] + floor);
	//      }

	ElemType aveMultiplier = 0, a;
	for (long i = 0; i < n; i++)
	{
		avars[i] = RMS_GAMMA * avars[i] + ONE_MINUS_GAMMA * (curr_grad[i] * curr_grad[i]);
		const int grad_sign = (ElemType(0) < curr_grad[i]) - (curr_grad[i] < ElemType(0));

		if (signs[i] * grad_sign > 0) steps[i] = std::min(steps[i] * RMS_WGT_INC, RMS_WGT_MAX);
		else steps[i] = std::max(steps[i] * RMS_WGT_DEC, RMS_WGT_MIN);

		a = steps[i] / sqrt(avars[i] + floor);
		curr_grad[i] *= a;
		signs[i] = (ElemType) grad_sign;
		if (needAveMultiplier) aveMultiplier += a;
	}
	if (needAveMultiplier) return aveMultiplier / n;
	return 1;
}

template <class ElemType>
template <typename GradType>
void CPUMatrix<ElemType>::AdaDelta(CPUMatrix<GradType>& gradients, CPUMatrix<ElemType>& functionValues, ElemType learningRate, ElemType rho, ElemType epsilon)
{
	size_t numColsNeeded = 2 * gradients.GetNumCols();

	if (IsEmpty() || (GetNumCols() < numColsNeeded))
	{
		Reset();
		Resize(gradients.GetNumRows(), numColsNeeded);
	}
	if (GetNumRows() != gradients.GetNumRows() || GetNumCols() != numColsNeeded)
		LogicError("The matrix gradients does not have expected dimensions");

	size_t n = gradients.GetNumElements();
	GradType* grad = gradients.GetData();
	ElemType* smoothAda = GetData();
	ElemType* smoothX2 = GetData() + n;
	ElemType* val = functionValues.GetData();

	// TODO: Unroll 4-times for better performance leveraging vectorization
#pragma omp parallel for
	for (long i = 0; i < n; i++)
	{
		ElemType g = (ElemType)grad[i];
		ElemType adaSqr = rho * smoothAda[i] + (1 - rho) * g * g;
		smoothAda[i] = adaSqr;
		ElemType x2 = smoothX2[i];
		ElemType deltaX = -sqrt(x2 + epsilon) / sqrt(adaSqr + epsilon) * g;
		smoothX2[i] = rho * smoothX2[i] + (1 - rho) * deltaX * deltaX;
		val[i] += learningRate * deltaX;
	}
}

template <class ElemType>
void CPUMatrix<ElemType>::AdaDeltaFlushTimestamps(size_t cols, ElemType rho, int* timestamps, int currentTimestamp)
{
	// Sets all timestamps to 0 and updates the two logical buffers that this object holds
	// so that their values are the same as if a dense implementation of adadelta had been used.
	// This basically means that the values of these buffers are set to decay * original value 
	// where decay is rho ** (currentTimestamp - timestamp for that column)
	auto rows = GetNumRows();
	auto smoothAda = GetData();
	auto smoothX2 = GetData() + cols * rows;
#pragma omp parallel for
	for (auto col = 0; col < cols; ++col)
	{
		ElemType decay = std::pow(rho, ElemType(currentTimestamp - timestamps[col]));
		auto offset = rows * col;
		timestamps[col] = 0;
		for (auto row = 0; row < rows; ++row)
		{
			smoothAda[offset + row] *= decay;
			smoothX2[offset + row] *= decay;
		}
	}
}

// allocated by the callee but should be deleted by the caller
// TODO: change to use STL vector instead
///template <class ElemType>
///ElemType* CPUMatrix<ElemType>::CopyToArray() const
///{
///	size_t numElements = GetNumElements();
///	if (numElements != 0)
///	{
///		//ElemType* arrayCopyTo = NewArray<ElemType>(numElements);
///		ElemType* arrayCopyTo = new ElemType[numElements];
///		memcpy(arrayCopyTo, GetData(), sizeof(ElemType) * numElements);
///		return arrayCopyTo;
///	}
///	return nullptr;
///}

////memory will be allocated by the callee if not enough but need to be deleted by the caller after it's done
////return number of elements copied
///template <class ElemType>
///size_t CPUMatrix<ElemType>::CopyToArray(ElemType*& copyTo, size_t& currSize) const
///{
///	size_t n = GetNumElements();
///	if (n > currSize) { delete[] copyTo; copyTo = new ElemType[currSize=n]; }
///	if (n) memcpy(copyTo, GetData(), n*sizeof(ElemType));
///	return n;
///}

///template <typename ElemType>
///void CPUMatrix<ElemType>::CopySection(size_t numRows, size_t numCols, ElemType* dst, size_t colStride) const
///{
///	size_t nr = min(numRows,m_numRows); if (nr==0) return;
///	size_t nc = min(numCols,m_numCols); if (nc==0) return;
///	ElemType* src = GetData();
///	for (size_t j=0; j<nc; ++j)
///	{
///		memcpy(dst, src, nr*sizeof(ElemType));
///		dst += colStride; src += GetNumRows();
///	}
///}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//			math functions
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::operator+=(ElemType alpha)
{
	return AssignSumOf(alpha, *this);
}

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::operator+(ElemType alpha) const
{
	CPUMatrix<ElemType> c(GetNumRows(), GetNumCols());
	c.AssignSumOf(alpha, *this);
	return c;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignSumOf(ElemType alpha, const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty())
		LogicError("AssignSumOf; Matrix a is empty");

	auto& us = *this;
	if (this != &a) Resize(a.GetNumRows(), a.GetNumCols());
	long m = (long) GetNumRows(), n = (long) GetNumCols();
	long m4 = m & ~3;
#pragma omp parallel for
	for (long j = 0; j < n; j++)
	{
		// four-way unrolling
		for (long i = 0; i < m4; i += 4)
		{
			us(i, j) = alpha + a(i, j);
			us(i+1, j) = alpha + a(i+1, j);
			us(i+2, j) = alpha + a(i+2, j);
			us(i+3, j) = alpha + a(i+3, j);
		}
		// handle remaining stuffs
		for (long i = m4; i < m; i++)
			us(i, j) = alpha + a(i, j);
	}
	return *this;
}

// if [this] and a have same dimension then [this]=[this]+a
// if a is a column vector, add to all columns of [this]
// if a is a row vector, add to all rows of [this]
// if a is a scalar, add it to all elements.
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::operator+=(const CPUMatrix<ElemType>& a)
{
	// if (a.GetNumElements() == 1)
	//    *this += a(0,0);
	// else
	ScaleAndAdd(1, a, *this);

	return *this;
}

// if [this] and a have same dimension then OUTPUT=[this]+a
// if a is a column vector, add to all columns of [this]
// if a is a row vector, add to all rows of [this]
template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::operator+(const CPUMatrix<ElemType>& a) const
{
	if (GetNumElements() == 1) { CPUMatrix<ElemType> c(a); c += *GetData(); return c; }
	if (a.GetNumElements() == 1) { CPUMatrix<ElemType> c(*this); c += *a.GetData(); return c; }
	// this implementation will introduce a copy overhead. but make resue of the code
	CPUMatrix<ElemType> c(*this); c += a;
	return c;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignSumOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b)
{
	if (a.GetNumElements() == 1) AssignSumOf(*a.GetData(), b); //{ SetValue(b); (*this) += a; }
	else { SetValue(a); (*this) += b; }
	return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::operator-=(ElemType alpha)
{
	return AssignDifferenceOf(*this, alpha);
}

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::operator-(ElemType alpha) const
{
	CPUMatrix<ElemType> c(GetNumRows(), GetNumCols());
	c.AssignDifferenceOf(*this, alpha);
	return c;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignDifferenceOf(ElemType alpha, const CPUMatrix<ElemType>& a)
{
	auto& us = *this;
	if (this != &a) Resize(a.GetNumRows(), a.GetNumCols());
	long m = (long) GetNumRows(), n = (long) GetNumCols();
	long m4 = m & ~3;
#pragma omp parallel for
	for (long j = 0; j < n; j++)
	{
		// four-way unrolling
		for (long i = 0; i < m4; i += 4)
		{
			us(i, j) = alpha - a(i, j);
			us(i+1, j) = alpha - a(i+1, j);
			us(i+2, j) = alpha - a(i+2, j);
			us(i+3, j) = alpha - a(i+3, j);
		}
		// handle remaining stuffs
		for (long i = m4; i < m; i++)
		{
			us(i, j) = alpha - a(i, j);
		}
	}
	return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignDifferenceOf(const CPUMatrix<ElemType>& a, ElemType alpha)
{
	auto& us = *this;
	if (this != &a) Resize(a.GetNumRows(), a.GetNumCols());
	long m = (long) GetNumRows(), n = (long) GetNumCols();
	long m4 = m & ~3;
#pragma omp parallel for
	for (long j = 0; j < n; j++)
	{
		// four-way unrolling
		for (long i = 0; i < m4; i += 4)
		{
			us(i, j) = a(i, j) - alpha;
			us(i+1, j) = a(i+1, j) - alpha;
			us(i+2, j) = a(i+2, j) - alpha;
			us(i+3, j) = a(i+3, j) - alpha;
		}
		// handle remaining stuffs
		for (long i = m4; i < m; i++)
		{
			us(i, j) = a(i, j) - alpha;
		}
	}
	return *this;
}

// if [this] and a have same dimension then [this]=[this]-a
// if a is a column vector, minus it from all columns of [this]
// if a is a row vector, minus it from all rows of [this]
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::operator-=(const CPUMatrix<ElemType>& a)
{
	ScaleAndAdd(-1, a, *this);
	return *this;
}

// if [this] and a have same dimension then output=[this]-a
// if a is a column vector, minus it from all columns of [this]
// if a is a row vector, minus it from all rows of [this]
template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::operator-(const CPUMatrix<ElemType>& a) const
{
	CPUMatrix<ElemType> c(*this); c -= a;
	return c;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignDifferenceOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b)
{
	if (this != &a) { Resize(a.GetNumRows(), a.GetNumCols()); SetValue(a); }
	(*this) -= b;
	return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::operator*=(ElemType alpha)
{
	Scale(alpha, *this);
	return *this;
}

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::operator*(ElemType alpha) const
{
	CPUMatrix<ElemType> c(GetNumRows(), GetNumCols());
	Scale(alpha, *this, c);
	return c;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignProductOf(ElemType alpha, const CPUMatrix<ElemType>& a)
{
	Scale(alpha, a, *this);
	return *this;
}

// [this] = a * b
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignProductOf(const CPUMatrix<ElemType>& a, bool transposeA, const CPUMatrix<ElemType>& b, bool transposeB)
{
	if (a.GetNumElements() == 1) { if (transposeB) AssignTransposeOf(b); (*this) *= a.GetData()[0]; }
	else if (b.GetNumElements() == 1) { if (transposeA) AssignTransposeOf(a); (*this) *= b.GetData()[0]; }
	else Multiply(a, transposeA, b, transposeB, *this);
	return *this;
}

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::operator*(const CPUMatrix<ElemType>& a) const
{
	if (GetNumElements() == 1) { CPUMatrix<ElemType> c; c.AssignProductOf(GetData()[0], a); return c; }
	if (a.GetNumElements() == 1) { CPUMatrix<ElemType> c; c.AssignProductOf(a.GetData()[0], *this); return c; }
	CPUMatrix<ElemType> c; Multiply(*this, a, c);
	return c;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::operator/=(ElemType alpha)
{
	(*this) *= 1 / alpha;
	return (*this);
}

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::operator/(ElemType alpha) const
{
	return ((*this) * ElemType(1 / alpha));
}

// element-wise power
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::operator^=(ElemType alpha)
{
	ElementWisePower(alpha, *this, *this);
	return *this;
}

//element-wise power
template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::operator^(ElemType alpha) const
{
	CPUMatrix<ElemType> c(GetNumRows(), GetNumCols());
	ElementWisePower(alpha, *this, c);
	return c;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignElementPowerOf(const CPUMatrix<ElemType>& a, ElemType power)
{
	ElementWisePower(power, a, *this);
	return *this;
}

// [this] = [this] .* a (we cannot override operator .* in c++)
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::ElementMultiplyWith(const CPUMatrix<ElemType>& a)
{
	return AssignElementProductOf(*this, a);
}

// [this] = a .* b
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignElementProductOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b)
{
	if (a.IsEmpty() || b.IsEmpty())
		LogicError("AssignElementProductOf; A[%lu,%lu] or B[%lu,%lu] is empty", a.GetNumRows(), a.GetNumCols(), b.GetNumRows(), b.GetNumCols());

	if (a.GetNumRows()!=b.GetNumRows() || a.GetNumCols()!=b.GetNumCols())
		LogicError("AssignElementProductOf; A[%lu,%lu] and B[%lu,%lu] have different size", a.GetNumRows(), a.GetNumCols(), b.GetNumRows(), b.GetNumCols());

	if (this != &a) Resize(a.GetNumRows(), a.GetNumCols());
	long m = (long) GetNumRows();
	long n = (long) GetNumCols();
	long m4 = m & ~3;
#pragma omp parallel for
	for (long j = 0; j < n; j++)
	{
		const ElemType* pa = a.GetDataCol(j);
		const ElemType* pb = b.GetDataCol(j);
		ElemType* po = GetDataCol(j);
		for (long i = 0; i < m4; i += 4)
		{
			po[i] = pa[i] * pb[i];
			po[i+1] = pa[i+1] * pb[i+1];
			po[i+2] = pa[i+2] * pb[i+2];
			po[i+3] = pa[i+3] * pb[i+3];
		}
		// handle remaining stuffs
		for (long i = m4; i < m; i++) po[i] = pa[i] * pb[i];
	}
	return *this;
}

// [this] += a .* b
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AddElementProductOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b)
{
	if (a.IsEmpty() || b.IsEmpty())
		LogicError("AssignElementProductOf; A[%lu,%lu] or B[%lu,%lu] is empty", a.GetNumRows(), a.GetNumCols(), b.GetNumRows(), b.GetNumCols());

	if (a.GetNumRows()!=b.GetNumRows() || a.GetNumCols()!=b.GetNumCols())
		LogicError("AssignElementProductOf; A[%lu,%lu] and B[%lu,%lu] have different size", a.GetNumRows(), a.GetNumCols(), b.GetNumRows(), b.GetNumCols());

	if (a.GetNumRows()!=GetNumRows() || a.GetNumCols()!=GetNumCols())
		LogicError("AssignElementProductOf; A[%lu,%lu] and C[%lu,%lu] have different size", a.GetNumRows(), a.GetNumCols(), GetNumRows(), GetNumCols());

	long m = (long) GetNumRows();
	long n = (long) GetNumCols();
	long m4 = m & ~3;
#pragma omp parallel for
	for (long j = 0; j < n; j++)
	{
		const ElemType* pa = a.GetDataCol(j);
		const ElemType* pb = b.GetDataCol(j);
		ElemType* po = GetDataCol(j);
		for (long i = 0; i < m4; i += 4)
		{
			po[i] += pa[i] * pb[i];
			po[i+1] += pa[i+1] * pb[i+1];
			po[i+2] += pa[i+2] * pb[i+2];
			po[i+3] += pa[i+3] * pb[i+3];
		}
		// handle remaining stuffs
		for (long i = m4; i < m; i++) po[i] += pa[i] * pb[i];
	}
	return *this;
}

// [this] = a ./ b
// TODO: This clips the divisor by a small value. Is that really what one would want?
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignElementDivisionOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b)
{
	if (a.IsEmpty() || b.IsEmpty())
		LogicError("AssignElementDivisionOf; A[%lu,%lu] or B[%lu,%lu] is empty",
						a.GetNumRows(), a.GetNumCols(), b.GetNumRows(), b.GetNumCols());

	if (a.GetNumRows()!=b.GetNumRows() || a.GetNumCols()!=b.GetNumCols())
		InvalidArgument("AssignElementDivisionOf; A[%lu,%lu] and B[%lu,%lu] havedifferent size",
						a.GetNumRows(), a.GetNumCols(), b.GetNumRows(), b.GetNumCols());

	auto& us = *this;
	if (this != &a) Resize(a.GetNumRows(), a.GetNumCols());
	ElemType smallValue = EPS_IN_INVERSE;

#pragma omp parallel for
	foreach_coord (i, j, us)
	{
		ElemType v = b(i, j);
		if (v >= 0 && v < smallValue) us(i, j) = a(i, j) / smallValue;
		else if (v < 0 && v > -smallValue) us(i, j) = a(i, j) / (-smallValue);
		else us(i, j) = a(i, j) / v;
	}
	return *this;
}

// [this] = [this] ./ a
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::ElementDivideBy(const CPUMatrix<ElemType>& a)
{
	return AssignElementDivisionOf(*this, a);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::ColumnElementMultiplyWith(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty() || IsEmpty())
		LogicError("ColumnElementMultiplyWith; Matrix is empty");

	if (!(a.GetNumRows() == GetNumRows() && a.GetNumCols() == 1))
		InvalidArgument("ColumnElementMultiplyWith; The input matrix should be a col vector and match [this]'s rows");

	auto& us = *this;

	long m = (long) GetNumRows(), n = (long) GetNumCols();
	long m4 = m & ~3;
#pragma omp parallel for
	for (long j = 0; j < n; j++)
	{
		// four-way unrolling
		for (long i = 0; i < m4; i += 4)
		{
			us(i, j) *= a(i, 0);
			us(i+1, j) *= a(i+1, 0);
			us(i+2, j) *= a(i+2, 0);
			us(i+3, j) *= a(i+3, 0);
		}
		// handle remaining stuffs
		for (long i = m4; i < m; i++)
		{
			us(i, j) *= a(i, 0);
		}
	}

	return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::RowElementMultiplyWith(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty() || IsEmpty())
		LogicError("RowElementMultiplyWith; Matrix is empty");

	if (!(a.GetNumRows() == 1 && a.GetNumCols() == GetNumCols()))
		InvalidArgument("RowElementMultiplyWith; The input matrix should be a row vector and match [this]'s columns");

	auto& us = *this;

	long m = (long) GetNumRows(), n = (long) GetNumCols();
	long m4 = m & ~3;
#pragma omp parallel for
	for (long j = 0; j < n; j++)
	{
		ElemType v = a(0, j);
		// four-way unrolling
		for (long i = 0; i < m4; i += 4)
		{
			us(i, j) *= v;
			us(i+1, j) *= v;
			us(i+2, j) *= v;
			us(i+3, j) *= v;
		}
		// handle remaining stuffs
		for (long i = m4; i < m; i++)
		{
			us(i, j) *= v;
		}
	}

	return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::RowElementDivideBy(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty() || IsEmpty())
		LogicError("RowElementDivideBy; Matrix is empty");

	if (!(a.GetNumRows() == 1 && a.GetNumCols() == GetNumCols()))
		InvalidArgument("RowElementDivideBy; The input matrix should be a row vector and match [this]'s columns");

	auto& us = *this;

	long m = (long) GetNumRows(), n = (long) GetNumCols();
	long m4 = m & ~3;
#pragma omp parallel for
	for (long j = 0; j < n; j++)
	{
		ElemType v = a(0, j);
		if (v >= 0 && v < EPS_IN_INVERSE) v = EPS_IN_INVERSE;
		else if (v < 0 && v > -EPS_IN_INVERSE) v = (-EPS_IN_INVERSE);

		// four-way unrolling
		for (long i = 0; i < m4; i += 4)
		{
			us(i, j) /= v;
			us(i+1, j) /= v;
			us(i+2, j) /= v;
			us(i+3, j) /= v;
		}
		// handle remaining stuffs
		for (long i = m4; i < m; i++)
		{
			us(i, j) /= v;
		}
	}

	return *this;
}
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::ColumnElementDivideBy(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty() || IsEmpty())
		LogicError("ColumnElementDivideBy; Matrix is empty");

	if (!(a.GetNumRows() == GetNumRows() && a.GetNumCols() == 1))
		InvalidArgument("ColumnElementDivideBy; The input matrix should be a col vector and match [this]'s rows");

	auto& us = *this;

	long m = (long) GetNumRows(), n = (long) GetNumCols();

	ElemType smallValue = EPS_IN_INVERSE;
#pragma omp parallel for
	for (long j = 0; j < n; j++)
	{
		for (long i = 0; i < m; i++)
		{
			ElemType v = a(i, 0);
			if (v >= 0 && v < smallValue) us(i, j) /= smallValue;
			else if (v < 0 && v > -smallValue) us(i, j) /= (-smallValue);
			else us(i, j) /= v;
		}
	}

	return *this;
}

// [this] = 1 ./ a
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::ElementInverse()
{
	return AssignElementInverseOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignElementInverseOf(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty()) LogicError("AssignElementInverseOf; A[%lu,%lu] is empty", a.GetNumRows(), a.GetNumCols());

	auto& us = *this;
	if (this != &a) Resize(a.GetNumRows(), a.GetNumCols());
	ElemType smallValue = EPS_IN_INVERSE;

#pragma omp parallel for
	foreach_coord (i, j, us)
	{
		if (a(i, j) < 0 && a(i, j) > -smallValue) us(i, j) = 1 / (-smallValue);
		else if (a(i, j) >= 0 && a(i, j) < smallValue) us(i, j) = 1 / smallValue;
		else us(i, j) = 1 / a(i, j);
	}
	return *this;
}

// [this] = sigmoid([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceSigmoid()
{
	return AssignSigmoidOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignSigmoidOf(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty())
		LogicError("AssignSigmoidOf; Matrix a is empty");

	auto& us = *this;
	if (this != &a) Resize(a.GetNumRows(), a.GetNumCols());

#pragma omp parallel for
	foreach_coord (i, j, us)
	{
		if (a(i, j) >= 0) us(i, j) = 1 / (1 + exp(-a(i, j)));
		else { ElemType v = exp(a(i, j)); us(i, j) = v / (1 + v); }
	}
	return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceLinearRectifierDerivative()
{
	return AssignLinearRectifierDerivativeOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignLinearRectifierDerivativeOf(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty())
		LogicError("AssignLinearRectifierDerivativeOf; Matrix a is empty");

	auto& us = *this;
	if (this != &a) Resize(a.GetNumRows(), a.GetNumCols());

	long m = (long) GetNumRows(), n = (long) GetNumCols();
	long m4 = m & ~3;
#pragma omp parallel for
	for (long j = 0; j < n; j++)
	{
		// four-way unrolling
		for (long i = 0; i < m4; i += 4)
		{
			us(i, j) = a(i, j) > 0.0f ? 1.0f : 0.0f;
			us(i+1, j) = a(i+1, j) > 0.0f ? 1.0f : 0.0f;
			us(i+2, j) = a(i+2, j) > 0.0f ? 1.0f : 0.0f;
			us(i+3, j) = a(i+3, j) > 0.0f ? 1.0f : 0.0f;
		}
		// handle remaining stuffs
		for (long i = m4; i < m; i++)
		{
			us(i, j) = a(i, j) > 0.0f ? 1.0f : 0.0f;
		}
	}
	return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceSigmoidDerivative()
{
	return AssignSigmoidDerivativeOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignSigmoidDerivativeOf(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty())
		LogicError("AssignSigmoidDerivativeOf; Matrix a is empty");

	auto& us = *this;
	if (this != &a) Resize(a.GetNumRows(), a.GetNumCols());

	long m = (long) GetNumRows(), n = (long) GetNumCols();
	long m4 = m & ~3;
#pragma omp parallel for
	for (long j = 0; j < n; j++)
	{
		// four-way unrolling
		for (long i = 0; i < m4; i += 4)
		{
			ElemType v = a(i, j); us(i, j) = v * (1 - v);
			ElemType v1 = a(i+1, j); us(i+1, j) = v1 * (1 - v1);
			ElemType v2 = a(i+2, j); us(i+2, j) = v2 * (1 - v2);
			ElemType v3 = a(i+3, j); us(i+3, j) = v3 * (1 - v3);
		}
		// handle remaining stuffs
		for (long i = m4; i < m; i++)
		{
			ElemType v = a(i, j);
			us(i, j) = v * (1 - v);
		}
	}
	return *this;
}

// [this] = tanh([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceTanh()
{
	return AssignTanhOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignTanhOf(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty())
		LogicError("AssignTanhOf; Matrix a is empty");

	auto& us = *this;
	if (this != &a) Resize(a.GetNumRows(), a.GetNumCols());

	long m = (long) GetNumRows(), n = (long) GetNumCols();
	long m4 = m & ~3;
#pragma omp parallel for
	for (long j = 0; j < n; j++)
	{
		// four-way unrolling
		for (long i = 0; i < m4; i += 4)
		{
			us(i, j) = tanh(a(i, j));
			us(i+1, j) = tanh(a(i+1, j));
			us(i+2, j) = tanh(a(i+2, j));
			us(i+3, j) = tanh(a(i+3, j));
		}
		// handle remaining stuffs
		for (long i = m4; i < m; i++)
		{
			us(i, j) = tanh(a(i, j));
		}
	}

	return *this;
}

// [this] = atanh([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceAtanh()
{
	return AssignAtanhOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignAtanhOf(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty())
		LogicError("AssignAtanhOf; Matrix a is empty");

	auto& us = *this;
	if (this != &a) Resize(a.GetNumRows(), a.GetNumCols());

	long m = (long) GetNumRows(), n = (long) GetNumCols();
	long m4 = m & ~3;
#pragma omp parallel for
	for (long j = 0; j < n; j++)
	{
		// four-way unrolling
		for (long i = 0; i < m4; i += 4)
		{
			us(i, j) = atanh(a(i, j));
			us(i+1, j) = atanh(a(i+1, j));
			us(i+2, j) = atanh(a(i+2, j));
			us(i+3, j) = atanh(a(i+3, j));
		}
		// handle remaining stuffs
		for (long i = m4; i < m; i++)
		{
			us(i, j) = atanh(a(i, j));
		}
	}
	return *this;
}

// [this] = softmax([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceLogSoftmax(bool isColWise)
{
	return AssignLogSoftmaxOf(*this, isColWise);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignLogSoftmaxOf(const CPUMatrix<ElemType>& a, bool isColWise)
{
	if (a.IsEmpty())
		LogicError("AssignLogSoftmaxOf; Matrix a is empty");

	auto& us = *this;
	if (this != &a)
		Resize(a.GetNumRows(), a.GetNumCols());

	if (isColWise)
	{
#pragma omp parallel for
		foreach_column (j, a)
		{
			// we need to extract max before applying exp to avoid overflow
			ElemType maxV = a(0, j);
			foreach_row (i, a) maxV = std::max(maxV, a(i, j));

			ElemType sum = 0;
			foreach_row (i, a) sum += exp(us(i, j) = a(i, j) - maxV);
			sum = log(sum);
			foreach_row (i, us) us(i, j) -= sum;
		}
	}
	else
	{
#pragma omp parallel for
		foreach_row (i, a)
		{
			// we need to extract max before applying exp to avoid overflow
			ElemType maxV = a(i, 0);
			foreach_column (j, a) maxV = std::max(maxV, a(i, j));

			ElemType sum = 0;
			foreach_column (j, a) sum += exp(us(i, j) = a(i, j) - maxV);
			sum = log(sum);
			foreach_column (j, us) us(i, j) -= sum;
		}
	}

	return *this;
}

// [this] = hardmax([this])
//	the max element is 1 else is 0
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceHardmax(bool isColWise)
{
	return AssignHardmaxOf(*this, isColWise);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignHardmaxOf(const CPUMatrix<ElemType>& a, bool isColWise)
{
	if (a.IsEmpty())
		LogicError("AssignHardmaxOf; Matrix a is empty");

	auto& us = *this;
	if (this != &a) Resize(a.GetNumRows(), a.GetNumCols());
	bool isInplace = (us.GetData() == a.GetData());
	if (!isInplace) memset(us.GetData(), 0, a.GetNumElements() * sizeof(ElemType));

	if (isColWise)
	{
		foreach_column (j, a)
		{
			// we need to extract max
			ElemType maxV = a(0, j);
			long maxI = 0;
			foreach_row (i, a)
				if (maxV < a(i, j)) { maxV = a(i, j); maxI = i; }

			if (isInplace)
				memset(us.GetData() + j*a.GetNumRows(), 0, a.GetNumRows() * sizeof(ElemType));

			us(maxI, j) = 1.0f;
		}
	}
	else
	{
		foreach_row (i, a)
		{
			// we need to extract max
			ElemType maxV = a(i, 0);
			long maxJ = 0;
			foreach_column (j, a)
				if (maxV < a(i, j)) { maxV = a(i, j); maxJ = j; }
			if (isInplace)
			{
				foreach_column(j, us)
					us(i, j) = (j == maxJ) ? 1.0f : 0.0f;
			}
			else
				us(i, maxJ) = 1.0f;
		}
	}
	return *this;
}

// [this] = sqrt([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceSqrt()
{
	return AssignSqrtOf(*this);
}

//	to prevent negative values caused by floating operations, we force inputs to be >=0
//	this may, however, hide problems in the caller.
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignSqrtOf(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty())
		LogicError("AssignSqrtOf; Matrix a is empty");

	auto& us = *this;
	if (this != &a)
		Resize(a.GetNumRows(), a.GetNumCols());

	long m = (long) GetNumRows(), n = (long) GetNumCols();
	long m4 = m & ~3;
#pragma omp parallel for
	for (long j = 0; j < n; j++)
	{
		// four-way unrolling
		for (long i = 0; i < m4; i += 4)
		{
			us(i, j)     = sqrt(max((ElemType)0, a(i, j)));
			us(i+1, j) = sqrt(max((ElemType)0, a(i+1, j)));
			us(i+2, j) = sqrt(max((ElemType)0, a(i+2, j)));
			us(i+3, j) = sqrt(max((ElemType)0, a(i+3, j)));
		}
		// remaining
		for (long i = m4; i < m; i++)
		{
			us(i, j) = sqrt(max((ElemType)0, a(i, j)));
		}
	}
	return *this;
}

// [this] = exp([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceExp()
{
	return AssignExpOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignExpOf(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty())
		LogicError("AssignExpOf; Matrix a is empty");

	auto& us = *this;
	if (this != &a)
		Resize(a.GetNumRows(), a.GetNumCols());

	long m = (long) GetNumRows(), n = (long) GetNumCols();
	long m4 = m & ~3;
#pragma omp parallel for
	for (long j = 0; j < n; j++)
	{
		// four-way unrolling
		for (long i = 0; i < m4; i += 4)
		{
			us(i, j) = exp(a(i, j));
			us(i+1, j) = exp(a(i+1, j));
			us(i+2, j) = exp(a(i+2, j));
			us(i+3, j) = exp(a(i+3, j));
		}
		// handle remaining stuffs
		for (long i = m4; i < m; i++)
		{
			us(i, j) = exp(a(i, j));
		}
	}

	return *this;
}

// [this] = exp([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceAbs()
{
	return AssignAbsOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignAbsOf(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty())
		LogicError("AssignAbsOf; Matrix a is empty");

	auto& us = *this;
	if (this != &a)
		Resize(a.GetNumRows(), a.GetNumCols());

	long m = (long) GetNumRows(), n = (long) GetNumCols();
	long m4 = m & ~3;
#pragma omp parallel for
	for (long j = 0; j < n; j++)
	{
		// four-way unrolling
		for (long i = 0; i < m4; i += 4)
		{
			us(i, j) = abs(a(i, j));
			us(i+1, j) = abs(a(i+1, j));
			us(i+2, j) = abs(a(i+2, j));
			us(i+3, j) = abs(a(i+3, j));
		}
		// handle remaining stuffs
		for (long i = m4; i < m; i++)
		{
			us(i, j) = abs(a(i, j));
		}
	}

	return *this;
}

// [this] = log([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceLog()
{
	return AssignLogOf(*this);
}

// [this] = log([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceLog10()
{
	return AssignLog10Of(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignLogOf(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty())
		LogicError("AssignLogOf; Matrix a is empty");

	auto& us = *this;
	if (this != &a)
		Resize(a.GetNumRows(), a.GetNumCols());

#pragma omp parallel for
	foreach_coord (i, j, a)
	{
		const ElemType v = a(i, j);
		if (v < EPS_IN_LOG)
		{
			us(i, j) = LOG_EPS_IN_LOG;
		}
		else
			us(i, j) = log(v);
	}

	return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignLog10Of(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty())
		LogicError("AssignLogOf; Matrix a is empty");

	auto& us = *this;
	if (this != &a)
		Resize(a.GetNumRows(), a.GetNumCols());

#pragma omp parallel for
	foreach_coord (i, j, a)
	{
		const ElemType v = a(i, j);
		if (v <= 0)
			LogicError("AssignLogOf: Log can only applied to numbers larger than 0");
		else if (v < EPS_IN_LOG)
		{
			us(i, j) = LOG10_EPS_IN_LOG;
		}
		else
			us(i, j) = log10(v);
	}

	return *this;
}

// [this] = cos([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceCosine()
{
	return AssignCosineOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignCosineOf(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty())
		LogicError("AssignCosineOf; Matrix a is empty");

	auto& us = *this;
	if (this != &a)
		Resize(a.GetNumRows(), a.GetNumCols());

#pragma omp parallel for
	foreach_coord (i, j, a)
	{
		const ElemType v = a(i, j);
		us(i, j) = cos(v);
	}

	return *this;
}

// [this] = -sin([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceNegativeSine()
{
	return AssignNegativeSineOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignNegativeSineOf(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty())
		LogicError("AssignNegativeSineOf; Matrix a is empty");

	auto& us = *this;
	if (this != &a)
		Resize(a.GetNumRows(), a.GetNumCols());

#pragma omp parallel for
	foreach_coord (i, j, a)
	{
		const ElemType v = a(i, j);
		us(i, j) = -sin(v);
	}

	return *this;
}

// [this] = tan([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceTan()
{
	return AssignTanOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignTanOf(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty())
		LogicError("AssignTanOf; Matrix a is empty");

	auto& us = *this;
	if (this != &a)
		Resize(a.GetNumRows(), a.GetNumCols());

#pragma omp parallel for
	foreach_coord(i, j, a)
	{
		const ElemType v = a(i, j);
		us(i, j) = tan(v);
	}

	return *this;
}

// [this] = acos([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceAcos()
{
	return AssignAcosOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignAcosOf(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty())
		LogicError("AssignAcosOf; Matrix a is empty");

	auto& us = *this;
	if (this != &a)
		Resize(a.GetNumRows(), a.GetNumCols());

#pragma omp parallel for
	foreach_coord (i, j, a)
	{
		const ElemType v = a(i, j);
		us(i, j) = acos(v);
	}

	return *this;
}

// [this] = asin([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceAsin()
{
	return AssignAsinOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignAsinOf(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty())
		LogicError("AssignAsinOf; Matrix a is empty");

	auto& us = *this;
	if (this != &a)
		Resize(a.GetNumRows(), a.GetNumCols());

#pragma omp parallel for
	foreach_coord (i, j, a)
	{
		const ElemType v = a(i, j);
		us(i, j) = asin(v);
	}

	return *this;
}

// [this] = atan([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceAtan()
{
	return AssignAtanOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignAtanOf(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty())
		LogicError("AssignAtanOf; Matrix a is empty");

	auto& us = *this;
	if (this != &a) Resize(a.GetNumRows(), a.GetNumCols());
#pragma omp parallel for
	foreach_coord(i, j, a)
	{
		const ElemType v = a(i, j);
		us(i, j) = atan(v);
	}
	return *this;
}

// [this] = cosh([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceCosh()
{
	return AssignCoshOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignCoshOf(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty())
		LogicError("AssignCoshOf; Matrix a is empty");

	auto& us = *this;
	if (this != &a)
		Resize(a.GetNumRows(), a.GetNumCols());

#pragma omp parallel for
	foreach_coord (i, j, a)
	{
		const ElemType v = a(i, j);
		us(i, j) = cosh(v);
	}
	return *this;
}

// [this] = sinh([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceSinh()
{
	return AssignSinhOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignSinhOf(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty())
		LogicError("AssignSinhOf; Matrix a is empty");

	auto& us = *this;
	if (this != &a)
		Resize(a.GetNumRows(), a.GetNumCols());

#pragma omp parallel for
	foreach_coord (i, j, a)
	{
		const ElemType v = a(i, j);
		us(i, j) = sinh(v);
	}
	return *this;
}

// [this] = asinh([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceAsinh()
{
	return AssignAsinhOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignAsinhOf(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty())
		LogicError("AssignAsinhOf; Matrix a is empty");

	auto& us = *this;
	if (this != &a)
		Resize(a.GetNumRows(), a.GetNumCols());

#pragma omp parallel for
	foreach_coord (i, j, a)
	{
		const ElemType v = a(i, j);
		us(i, j) = asinh(v);
	}
	return *this;
}

//	Threshold truncating: this[i] = max( this[i], threshold )
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceTruncateBottom(ElemType threshold)
{
	if (IsEmpty())
		LogicError("InplaceTruncateBottom; Matrix is empty");

	auto& us = *this;
	long m = (long) GetNumRows(), n = (long) GetNumCols();
	long m4 = m & ~3;
#pragma omp parallel for
	for (long j = 0; j < n; j++)
	{
		// four-way unrolling
		for (long i = 0; i < m4; i += 4)
		{
			if (us(i, j) < threshold) us(i, j) = threshold;
			if (us(i+1, j) < threshold) us(i+1, j) = threshold;
			if (us(i+2, j) < threshold) us(i+2, j) = threshold;
			if (us(i+3, j) < threshold) us(i+3, j) = threshold;
		}
		// handle remaining stuffs
		for (long i = m4; i < m; i++)
			if (us(i, j) < threshold) us(i, j) = threshold;
	}

	return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceTruncate(ElemType threshold)
{
	if (IsEmpty())
		LogicError("InplaceTruncate; Matrix is empty");

	auto& us = *this;
	ElemType locThresholdPos = abs(threshold);
	ElemType locTHresholdNeg = -locThresholdPos;

	long m = (long) GetNumRows(), n = (long) GetNumCols();
	long m4 = m & ~3;
#pragma omp parallel for
	for (long j = 0; j < n; j++)
	{
		// four-way unrolling
		for (long i = 0; i < m4; i += 4)
		{
			if (us(i, j) > locThresholdPos) us(i, j) = locThresholdPos;
			else if (us(i, j) < locTHresholdNeg) us(i, j) = locTHresholdNeg;

			if (us(i+1, j) > locThresholdPos) us(i+1, j) = locThresholdPos;
			else if (us(i+1, j) < locTHresholdNeg) us(i+1, j) = locTHresholdNeg;

			if (us(i+2, j) > locThresholdPos) us(i+2, j) = locThresholdPos;
			else if (us(i+2, j) < locTHresholdNeg) us(i+2, j) = locTHresholdNeg;

			if (us(i+3, j) > locThresholdPos) us(i+3, j) = locThresholdPos;
			else if (us(i+3, j) < locTHresholdNeg) us(i+3, j) = locTHresholdNeg;
		}
		// handle remaining stuffs
		for (long i = m4; i < m; i++)
		{
			if (us(i, j) > locThresholdPos) us(i, j) = locThresholdPos;
			else if (us(i, j) < locTHresholdNeg) us(i, j) = locTHresholdNeg;
		}
	}
	return *this;
}

//		{ x - threshold		if x>threshold
//	x = { x + threshold		if x<-threshold
//		{ 0					otherwise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceSoftThreshold(ElemType threshold)
{
	if (IsEmpty())
		LogicError("InplaceTruncate; Matrix is empty");

	ElemType* ptr = GetData();
	long m = (long) GetNumElements();
	long m4 = m & ~3;
#pragma omp parallel for
	for (long i = 0; i < m4; i += 4) // four-way unrolling
	{
		if (ptr[i] > threshold) ptr[i] -= threshold;
		else if (ptr[i] < -threshold) ptr[i] += threshold;
		else ptr[i] = 0;

		if (ptr[i+1] > threshold) ptr[i+1] -= threshold;
		else if (ptr[i+1] < -threshold) ptr[i+1] += threshold;
		else ptr[i+1] = 0;

		if (ptr[i+2] > threshold) ptr[i+2] -= threshold;
		else if (ptr[i+2] < -threshold) ptr[i+2] += threshold;
		else ptr[i+2] = 0;

		if (ptr[i+3] > threshold) ptr[i+3] -= threshold;
		else if (ptr[i+3] < -threshold) ptr[i+3] += threshold;
		else ptr[i+3] = 0;
	}
	// handle remaining stuffs
	for (long i = m4; i < m; i++)
	{
		if (ptr[i] > threshold) ptr[i] -= threshold;
		else if (ptr[i] < -threshold) ptr[i] += threshold;
		else ptr[i] = 0;
	}
	return *this;
}

// Threshold truncating: this[i] = max( a[i], threshold )
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignTruncateBottomOf(const CPUMatrix<ElemType>& a, ElemType threshold)
{
	if (a.IsEmpty())
		LogicError("AssignTruncateBottomOf; Matrix a is empty");

	auto& us = *this;
	if (this != &a) Resize(a.GetNumRows(), a.GetNumCols());
#pragma omp parallel for
	foreach_coord (i, j, a)
	{
		if (a(i, j) < threshold) us(i, j) = threshold;
		else us(i, j) = a(i, j);
	}
	return *this;
}

//Threshold truncating: this[i] = min( this[i], threshold )
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceTruncateTop(ElemType threshold)
{
	if (IsEmpty())
		LogicError("InplaceTruncateTop; Matrix is empty");

	auto& us = *this;
#pragma omp parallel for
	foreach_coord (i, j, us)
		if (us(i, j) > threshold) us(i, j) = threshold;

	return *this;
}

//Threshold truncating: this[i] = min( a[i], threshold )
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignTruncateTopOf(const CPUMatrix<ElemType>& a, ElemType threshold)
{
	if (a.IsEmpty())
		LogicError("AssignTruncateTopOf; Matrix a is empty");

	auto& us = *this;
	if (this != &a)
		Resize(a.GetNumRows(), a.GetNumCols());

#pragma omp parallel for
	foreach_coord (i, j, a)
	{
		if (a(i, j) > threshold)
			us(i, j) = threshold;
		else
			us(i, j) = a(i, j);
	}

	return *this;
}

////Threshold truncating: this[i] = 0 if abs(this[i]<threshold).

///template <class ElemType>
///CPUMatrix<ElemType>& CPUMatrix<ElemType>::SetToZeroIfAbsLessThan(ElemType threshold)
///{
///	if (IsEmpty())
///		LogicError("SetToZeroIfAbsLessThan; Matrix is empty");

///	auto& us = *this;
///#pragma omp parallel for
///	foreach_coord (i, j, us)
///	{
///		if (abs(us(i, j)) < threshold) us(i, j) = 0;
///	}
///	return *this;
///}

//sum of all abs(elements)
template <class ElemType>
ElemType CPUMatrix<ElemType>::SumOfAbsElements() const
{
	if (IsEmpty())
		LogicError("SumOfAbsElements; Matrix is empty");

	if (std::is_same<ElemType, double>::value)
	{
		return (ElemType) cblas_dasum((int)GetNumElements(), reinterpret_cast<double*>(GetData()), 1);
	}
	else if (std::is_same<ElemType, float>::value)
	{
#pragma warning(suppress : 4244)
		return cblas_sasum((int)GetNumElements(), reinterpret_cast<float*>(GetData()), 1);
	}
	else
	{
		RuntimeError("SumOfAbsElements; Unsupported data type");
	}
}

//sum of all elements
template <class ElemType>
ElemType CPUMatrix<ElemType>::SumOfElements() const
{
	if (IsEmpty())
		LogicError("SumOfElements; Matrix is empty");

	ElemType sum = 0;
	long m = (long) GetNumElements(); // note: OpenMP requires loop indices to be long, not size_t
	long m4 = m & ~3;

	ElemType* bufPtr = GetData();
//four-way unrolling
#pragma omp parallel for reduction(+ : sum)
	for (long i = 0; i < m4; i += 4)
	{
		sum += bufPtr[i] + bufPtr[i+1] + bufPtr[i+2] + bufPtr[i+3];
	}
	// handle remaining stuffs
	for (long i = m4; i < m; i++)
	{
		sum += bufPtr[i];
	}

	return sum;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignSumOfElements(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty())
		LogicError("AssignSumOfElements; Matrix a is empty");

	Resize(1, 1);
	*GetData() = a.SumOfElements();
	return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignOneHot(const CPUMatrix<ElemType>& a, vector<size_t>& shape, size_t axis)
{
	if (a.IsEmpty())
		LogicError("AssignOneHot; Matrix a is empty");

	if (axis >= shape.size())
		LogicError("AssignOneHot: axis is not correct");
	size_t item_size = 1;
	for (size_t i = 0; i < shape.size() && i < axis; i++)
		item_size *= shape[i];

	size_t num_class = shape[axis];

	auto& us = *this;
	auto nCols = a.GetNumCols();
	auto nRows = num_class * a.GetNumRows();
	Resize(nRows, nCols);
	ElemType* bufPtr = GetData();
	ElemType* aBufPtr = a.GetData();
	memset(bufPtr, 0, sizeof(ElemType) * nRows *nCols);
#pragma omp parallel for
	for (long i = 0; i < a.GetNumElements(); i++)
	{
		if (aBufPtr[i] >= 0 && aBufPtr[i] < num_class)
		{
			size_t block_id = i / item_size;
			size_t item_id = i % item_size;
			bufPtr[block_id * num_class * item_size + item_id + item_size * (size_t)aBufPtr[i]] = 1;
		}
	}

	return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::GatherFromTarget(const CPUMatrix<ElemType>& indices, const CPUMatrix<ElemType>& target, size_t row_elements)
{
	if (indices.IsEmpty() || target.IsEmpty())
		LogicError("GatherFromTarget: input matrix is empty");

	if (row_elements == 0)
		LogicError("GatherFromTarget: target matrix at least need 1 dim");

	auto nCols = indices.GetNumCols();
	auto nRows = indices.GetNumRows() * row_elements;
	Resize(nRows, nCols);

	ElemType* indicesBufPtr = indices.GetData();
	ElemType* targetBufPtr = target.GetData();
	ElemType* buffer = GetData();

#pragma omp parallel for
	for (int i = 0; i < indices.GetNumElements(); i++)
	{
		memcpy(buffer + i * row_elements, targetBufPtr + ((size_t)indicesBufPtr[i] * row_elements), sizeof(ElemType) * row_elements);
	}

	return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::ScatterToIndices(const CPUMatrix<ElemType>& values, const CPUMatrix<ElemType>& indices, size_t row_elements,
	const CPUMatrix<char>* mask/*= nullptr*/)
{
	if (indices.IsEmpty() || values.IsEmpty() || (mask && mask->IsEmpty()))
		LogicError("ScatterToIndices: input matrix is empty");
	if (mask && (indices.GetNumCols() % mask->GetNumCols() != 0))
		LogicError("ScatterAccordingIndices; The number of columns(%zu) of the matrix slice to be masked is not a multiple of the number of columns(%zu) of the mask slice.",
			indices.GetNumCols(), mask->GetNumCols());

	ElemType* indicesBufPtr = indices.GetData();
	ElemType* valueBufPtr = values.GetData();
	char* maskBufPtr = mask ? mask->GetData() : nullptr;
	ElemType* buffer = GetData();
	size_t numElemsPerMaskEntry = mask ? indices.GetNumCols() / mask->GetNumCols() * indices.GetNumRows() : 0;

	ScatterValues(indicesBufPtr, valueBufPtr, buffer, static_cast<ElemType>(1), indices.GetNumElements(), row_elements, GetNumCols(), maskBufPtr, numElemsPerMaskEntry);

	return *this;
}

///template <class ElemType>
///void CPUMatrix<ElemType>::VectorSum(const CPUMatrix<ElemType>& a, CPUMatrix<ElemType>& c, bool isColWise)
///{
///	if (a.IsEmpty())
///		LogicError("VectorSum:  Input matrix a is empty");

///	const int m = (int)a.GetNumRows();
///	const int n = (int)a.GetNumCols();

///	assert(m > 0 && n > 0); // converting from size_t to int may cause overflow

///	if (isColWise) // col-wise
///	{
///		c.Resize(1, n);

///#pragma omp parallel for
///		foreach_column (j, a)
///		{
///			ElemType v = 0;
///			foreach_row (i, a)
///			{
///#pragma omp atomic
///				v += a(i, j);
///			}
///			c(0, j) = v;
///		}
///	}
///	else
///	{
///		c.Resize(m, 1);

///#pragma omp parallel for
///		foreach_row (i, a)
///		{
///			ElemType v = 0;
///			foreach_column (j, a)
///			{
///#pragma omp atomic
///				v += a(i, j);
///			}
///			c(i, 0) = v;
///		}
///	}
///}

template <class ElemType>
void CPUMatrix<ElemType>::VectorNorm1(CPUMatrix<ElemType>& c, bool isColWise) const
{
	if (IsEmpty())
		LogicError("VectorNorm1; Matrix is empty");

	auto& us = *this;

	const int m = (int)us.GetNumRows();
	const int n = (int)us.GetNumCols();

	if (isColWise) // col-wise
	{
		c.Resize(1, n);

#pragma omp parallel for
		foreach_column (j, us)
		{
			ElemType v = 0;
			foreach_row (i, us)
			{
#pragma omp atomic
				v += abs(us(i, j));
			}
			c(0, j) = v;
		}
	}
	else
	{
		c.Resize(m, 1);

#pragma omp parallel for
		foreach_row (i, us)
		{
			ElemType v = 0;
			foreach_column (j, us)
			{
#pragma omp atomic
				v += abs(us(i, j));
			}
			c(i, 0) = v;
		}
	}
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignVectorNorm1of(CPUMatrix<ElemType>& a, bool isColWise)
{
	a.VectorNorm1(*this, isColWise);
	return *this;
}

template <class ElemType>
void CPUMatrix<ElemType>::VectorNorm2(CPUMatrix<ElemType>& c, bool isColWise) const
{
	if (IsEmpty())
		LogicError("VectorNorm2; Matrix is empty");

	auto& us = *this;

	const int m = (int)us.GetNumRows();
	const int n = (int)us.GetNumCols();

	ElemType* bufPtr = us.GetData();
	if (isColWise) // col-wise
	{
		c.Resize(1, n);

		if (std::is_same<ElemType, double>::value)
		{
#pragma omp parallel for
			foreach_column (j, c)
			{
				c(0, j) = (ElemType) cblas_dnrm2(m, reinterpret_cast<double*>(bufPtr + us.ColumnPos(j)), 1);
			}
		}
		else if(std::is_same<ElemType, float>::value)
		{
#pragma omp parallel for
			foreach_column (j, c)
			{
#pragma warning(suppress : 4244)
				c(0, j) = cblas_snrm2(m, reinterpret_cast<float*>(bufPtr + us.ColumnPos(j)), 1);
			}
		}
		else
		{
			RuntimeError("VectorNorm2; Unsupported data type");
		}
	}
	else
	{
		c.Resize(m, 1);

		if (std::is_same<ElemType, double>::value)
		{
#pragma omp parallel for
			foreach_row (i, c)
			{
				c(i, 0) = cblas_dnrm2(n, reinterpret_cast<double*>(bufPtr + i), m);
			}
		}
		else if (std::is_same<ElemType, float>::value)
		{
#pragma omp parallel for
			foreach_row (i, c)
			{
#pragma warning(suppress : 4244)
				c(i, 0) = cblas_snrm2(n, reinterpret_cast<float*>(bufPtr + i), m);
			}
		}
		else
		{
			RuntimeError("VectorNorm2; Unsupported data type");
		}
	}
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignVectorNorm2of(CPUMatrix<ElemType>& a, bool isColWise)
{
	a.VectorNorm2(*this, isColWise);
	return *this;
}

template <class ElemType>
void CPUMatrix<ElemType>::VectorNormInf(CPUMatrix<ElemType>& c, bool isColWise) const
{
	if (IsEmpty())
		LogicError("VectorNormInf; Matrix is empty");

	auto& us = *this;

	const int m = (int)us.GetNumRows();
	const int n = (int)us.GetNumCols();

	if (isColWise) // col-wise
	{
		c.Resize(1, n);

		// #pragma omp parallel for
		foreach_column (j, us)
		{
			ElemType v = 0;
			foreach_row (i, us)
			{
				v = std::max(v, fabs_(us(i, j)));
			}
			c(0, j) = v;
		}
	}
	else
	{
		c.Resize(m, 1);

		// #pragma omp parallel for
		foreach_row (i, us)
		{
			ElemType v = 0;
			foreach_column (j, us)
			{
				v = std::max(v, fabs_(us(i, j)));
			}
			c(i, 0) = v;
		}
	}
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignVectorNormInfOf(CPUMatrix<ElemType>& a, bool isColWise)
{
	a.VectorNormInf(*this, isColWise);
	return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignInnerProductOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, bool isColWise)
{
	InnerProduct(a, b, *this, isColWise);
	return *this;
}

//column-wise crossproduct
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignKhatriRaoProductOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b)
{
	if (a.IsEmpty() || b.IsEmpty())
		LogicError("AssignKhatriRaoProductOf; Matrix is empty");

	long cols = (long) a.GetNumCols();
	if (cols != b.GetNumCols())
		InvalidArgument("a.GetNumCols() != b.GetNumCols()");

	long rowsA = (long) a.GetNumRows();
	long rowsB = (long) b.GetNumRows();
	Resize(rowsA * rowsB, cols);

#ifdef __INTEL_COMPILER // TODO: check this
#pragma simd statement
#endif
#pragma omp parallel for
	for (long k = 0; k < cols; k++)
	{
		long jj = 0;
		for (long j = 0; j < rowsB; j++)
		{
			for (long i = 0; i < rowsA; i++)
			{
				(*this)(jj++, k) = a(i, k) * b(j, k);
			}
		}
	}
	return *this;
}

//column-wise reshaped product. Used to compute KhatriRaoProduct Gradient
//   this = reshape each column of a from (K1xK2,1) to (K1, K2)
//   if each column of a is not transposed, each (K1, K2) times each column of b (K2, frames).
//   the output is a (K1, frames) matrix
//   if each column of a is tranposed, each (K1, K2)^T times each column of b(K1, frames) and output is (K2, frames)
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AddColumnReshapeProductOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, bool transposeAColumn)
{
	if (a.IsEmpty() || b.IsEmpty())
		LogicError("AddColumnReshapeProductOf; Matrix is empty");

	long cols = (long) a.GetNumCols();
	if (cols != b.GetNumCols())
		InvalidArgument("AddColumnReshapeProductOf: a.GetNumCols() != b.GetNumCols()");

	long rowsA = (long) a.GetNumRows();
	long rowsB = (long) b.GetNumRows();

	if (rowsA % rowsB != 0)
		InvalidArgument("AddColumnReshapeProductOf: number of rows in a should be multiples of that in b");

	long rowsC = rowsA / rowsB;
	if (rowsC != GetNumRows() || cols != GetNumCols())
		InvalidArgument("AddColumnReshapeProductOf: This matrix does not have the right size");

	auto& us = *this;

	if (transposeAColumn)
	{
		// find nrows and ncols of tbe reshaped a
		long nrows = rowsB;
		long ncols = rowsC;

#ifdef __INTEL_COMPILER // TODO: check this
#pragma simd statement
#endif
#pragma omp parallel for
		foreach_column (t, a)
		{
			size_t k = 0;
			for (size_t j = 0; j < ncols; j++) // row and col is transposed
			{
				ElemType v = 0;
				for (size_t i = 0; i < nrows; i++)
				{
					v += a(k, t) * b(i, t);
					k++;
				}
				us(j, t) += v;
			}
		}
	}
	else
	{
		size_t ncols = rowsB;
		size_t nrows = rowsC;

#ifdef __INTEL_COMPILER // TODO: check this
#pragma simd statement
#endif
#pragma omp parallel for
		foreach_column (t, a)
		{
			size_t k = 0;
			for (size_t j = 0; j < ncols; j++)
			{
				for (size_t i = 0; i < nrows; i++)
				{
					us(i, t) += a(k, t) * b(j, t);
					k++;
				}
			}
		}
	}
	return *this;
}

///template <class ElemType>
///CPUMatrix<ElemType>& CPUMatrix<ElemType>::AddWithScaleOf(ElemType alpha, const CPUMatrix<ElemType>& a)
///{
///	ScaleAndAdd(alpha, a, *this);
///	return *this;
///}

template <class ElemType>
ElemType CPUMatrix<ElemType>::FrobeniusNorm() const
{
	if (IsEmpty())
		LogicError("FrobeniusNorm; Matrix is empty");

	ElemType v = 0;

	long m = (long) GetNumElements();
	long m4 = m & ~3;

	ElemType* bufPtr = GetData();
#pragma omp parallel for reduction(+ : v)
	for (long i = 0; i < m4; i += 4)
	{
		v += bufPtr[i] * bufPtr[i] + bufPtr[i+1] * bufPtr[i+1] + bufPtr[i+2] * bufPtr[i+2] + bufPtr[i+3] * bufPtr[i+3];
	}
	// handle remaining stuffs
	for (long i = m4; i < m; i++)
	{
		v += bufPtr[i] * bufPtr[i];
	}

	return sqrt(v);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignFrobeniusNormOf(const CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty())
		LogicError("AssignFrobeniusNormOf; Matrix a is empty");

	auto& us = *this;
	us.Resize(1, 1);
	us(0, 0) = a.FrobeniusNorm();

	return us;
}

template <class ElemType>
ElemType CPUMatrix<ElemType>::MatrixNormInf() const
{
	if (IsEmpty())
		LogicError("MatrixNormInf; Matrix is empty");

	auto& us = *this;

	ElemType v = 0;
#pragma omp parallel for
	foreach_coord (i, j, us)
	{
#pragma omp critical
		{
			v = std::max(v, fabs_(us(i, j)));
		}
	}
	return v;
}

template <class ElemType>
ElemType CPUMatrix<ElemType>::MatrixNorm0() const
{
	if (IsEmpty())
		LogicError("MatrixNorm0; Matrix is empty");

	auto& us = *this;

	ElemType v = 0;
#pragma omp parallel for
	foreach_coord (i, j, us)
	{
		if (us(i, j) != 0)
		{
#pragma omp critical
			{
				++v;
			}
		}
	}
	return v;
}

template <class ElemType>
ElemType CPUMatrix<ElemType>::MatrixNorm1() const
{
	if (IsEmpty())
		LogicError("MatrixNorm1; Matrix is empty");

	auto& us = *this;

	ElemType sum = 0;
#pragma omp parallel for reduction(+ : sum)
	foreach_coord (i, j, us)
	{
		sum += abs(us(i, j));
	}
	return sum;
}

///template <class ElemType>
///CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignSignOf(const CPUMatrix<ElemType>& a)
///{
///	if (a.IsEmpty())
///		LogicError("AssignSignOf; Matrix a is empty");

///	auto& us = *this;
///	if (this != &a)
///		Resize(a.GetNumRows(), a.GetNumCols());

///#pragma omp parallel for
///	foreach_column (j, us)
///	{
///		foreach_row (i, us)
///		{
///			ElemType v = a(i, j);
///			if (!std::isnan(v))
///				us(i, j) = (v == (ElemType) 0 ? (ElemType) 0 : (v > 0 ? (ElemType) 1 : (ElemType)(-1)));
///			else
///				us(i, j) = v;
///		}
///	}

///	return us;
///}

///template <class ElemType>
///CPUMatrix<ElemType>& CPUMatrix<ElemType>::AddSignOf(const CPUMatrix<ElemType>& a)
///{
///	if (a.IsEmpty())
///		LogicError("AddSignOf; Matrix a is empty");

///	auto& us = *this;
///	if (this != &a)
///		Resize(a.GetNumRows(), a.GetNumCols());

///#pragma omp parallel for
///	foreach_column (j, us)
///	{
///		foreach_row (i, us)
///		{
///			ElemType v = a(i, j);
///			if (!std::isnan(v))
///				us(i, j) += (v == (ElemType) 0 ? (ElemType) 0 : (v > 0 ? (ElemType) 1 : (ElemType)(-1)));
///			else
///				us(i, j) = v;
///		}
///	}

///	return us;
///}

// I decided to use CPUMatrix<ElemType>& maxIndexes instead of integer vector because the result may be used to do additional calculation
template <class ElemType>
void CPUMatrix<ElemType>::VectorMax(CPUMatrix<ElemType>& maxIndexes, CPUMatrix<ElemType>& maxValues, bool isColWise, int topK) const
{
	if (IsEmpty())
		LogicError("VectorMax; Matrix is empty");

	auto& us = *this;
	const int m = (int)GetNumRows();
	const int n = (int)GetNumCols();
	if (topK > m)
		InvalidArgument("VectorMax: TopK must be less or equal than the number of rows");

	assert(m > 0 && n > 0); // converting from size_t to int may cause overflow

	if (isColWise) // col-wise
	{
		maxValues.Resize(topK, n);
		maxIndexes.Resize(topK, n);

		if (topK == 1)
		{
#pragma omp parallel for
			for (int j = 0; j < n; j++)
			{
				ElemType v = us(0, j);
				size_t index = 0;
				foreach_row (i, us)
				{
					if (v < us(i, j))
					{
						index = i;
						v = us(i, j);
					}
				}
				maxValues(0, j) = v;
				maxIndexes(0, j) = (ElemType) index;
			}
		}
		else
		{
			std::vector<int> indices(m);

			const ElemType* curVal = GetData();
			ElemType* curIdx = maxIndexes.GetData();
			ElemType* curMax = maxValues.GetData();
			for (int icol = 0; icol < n; icol++, curVal += m, curIdx += topK, curMax += topK)
			{
				std::iota(indices.begin(), indices.end(), 0);
				// Partial sort, descending order.
				std::partial_sort(indices.begin(), indices.begin() + topK, indices.end(),
									[curVal](const int& a, const int& b)
									{
										return curVal[a] > curVal[b];
									});
				// REVIEW alexeyk: the following produces warning (see SCL_SECURE_NO_WARNINGS) so use loop instead.
				// std::transform(indices.begin(), indices.begin() + topK, curIdx, [](const int& a) { return static_cast<ElemType>(a); });
				for (int i2 = 0; i2 < topK; i2++)
				{
					curIdx[i2] = static_cast<ElemType>(indices[i2]);
					curMax[i2] = curVal[indices[i2]];
				}
			}
		}
	}
	else
	{
		if (topK > 1)
			RuntimeError("Row-wise TopK max is not supported");

		maxValues.Resize(m, 1);
		maxIndexes.Resize(m, 1);

#pragma omp parallel for
		for (int i = 0; i < m; i++)
		{
			ElemType v = us(i, 0);
			size_t index = 0;
			foreach_column (j, us)
			{
				if (v < us(i, j))
				{
					index = j;
					v = us(i, j);
				}
			}
			maxValues(i, 0) = v;
			maxIndexes(i, 0) = (ElemType) index;
		}
	}
}

template <class ElemType>
void CPUMatrix<ElemType>::VectorMin(CPUMatrix<ElemType>& minIndexes, CPUMatrix<ElemType>& minValues, bool isColWise) const
{
	if (IsEmpty())
		LogicError("VectorMin; Matrix is empty");

	auto& us = *this;
	const int m = (int)GetNumRows();
	const int n = (int)GetNumCols();

	assert(m > 0 && n > 0); // converting from size_t to int may cause overflow

	if (isColWise) // col-wise
	{
		minValues.Resize(1, n);
		minIndexes.Resize(1, n);

#pragma omp parallel for
		for (int j = 0; j < n; j++)
		{
			ElemType v = us(0, j);
			size_t index = 0;
			foreach_row (i, us)
			{
				if (v > us(i, j))
				{
					index = i;
					v = us(i, j);
				}
			}
			minValues(0, j) = v;
			minIndexes(0, j) = (ElemType) index;
		}
	}
	else
	{
		minValues.Resize(m, 1);
		minIndexes.Resize(m, 1);

#pragma omp parallel for
		for (int i = 0; i < m; i++)
		{
			ElemType v = us(i, 0);
			size_t index = 0;
			foreach_column (j, us)
			{
				if (v > us(i, j))
				{
					index = j;
					v = us(i, j);
				}
			}
			minValues(i, 0) = v;
			minIndexes(i, 0) = (ElemType) index;
		}
	}
}

///template <class ElemType>
///CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignNumOfDiff(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, bool searchInCol)
///{
///	if (a.GetNumCols() != b.GetNumCols())
///		throw std::invalid_argument("AssignNumOfDiff: a and b must have the same number of columns");
///	if (!searchInCol && a.GetNumRows() != b.GetNumRows())
///		throw std::invalid_argument("AssignNumOfDiff: a and b must have the same number of rows");

///	ElemType n = 0;
///	if (!searchInCol)
///	{
///		foreach_coord (i, j, a)
///		{
///			n += (a(i, j) != b(i, j));
///		}
///	}
///	else
///	{
///		size_t crow = b.GetNumRows();
///		const ElemType* curCol = b.GetData();
///		for (size_t icol = 0; icol < a.GetNumCols(); icol++, curCol += crow)
///		{
///			auto res = std::find(curCol, curCol + crow, a(0, icol));
///			if (res == curCol + crow)
///				n++;
///		}
///	}

///	Resize(1, 1); // result should be one element
///	(*this)(0, 0) = n;

///	return *this;
///}

//// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
////			Other helper Functions
//// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

///struct PrintRange
///{
///	// print from begin to skipBegin, then from skipEnd to end
///	// skipBegin = end if no split
///	size_t begin;
///	size_t skipBegin;
///	size_t skipEnd;
///	size_t end;
///	bool IsEmpty() const { return end <= begin; }

///	// examples:
///	//  * 3..10
///	//  * -3..-3: include end-3..end and 0..3
///	PrintRange(ptrdiff_t first, ptrdiff_t last, size_t total)
///	{
///		if (first >= 0 && last >= 0)
///		{
///			begin = (size_t)first;
///			end = (size_t)last + 1;
///			if (end > total)    // allow INT_MAX, meaning to end
///				end = total;
///			skipBegin = end;
///			skipEnd = end;
///		}
///		else if (first < 0 && last < 0)
///		{
///			begin = 0;
///			skipBegin = (size_t)(-last);
///			skipEnd = (size_t)(total + first);
///			if (skipEnd <= skipBegin)
///				skipBegin = skipEnd = total;
///			end = total;
///		}
///		else    // if other combinations are ever of interest then implement them here
///			LogicError("Print: Bounds must be either both positive or both negative");
///	}
///};

//// use negative ranges to print corners, e.g. Print("name", -3, -3, -3, -3) will print the first 3 and last 3 rows/cols
///template <class ElemType>
///void CPUMatrix<ElemType>::Print(const char* matrixName, ptrdiff_t rowFirst, ptrdiff_t rowLast, ptrdiff_t colFirst, ptrdiff_t colLast) const
///{
///	fprintf(stderr, "\n###### ");
///	if (matrixName != nullptr)
///		fprintf(stderr, "%s ", matrixName);
///	fprintf(stderr, "(%lu, %lu)", (unsigned long)GetNumRows(), (unsigned long)GetNumCols());
///	if (rowFirst != 0 || colFirst != 0 || (size_t)(rowLast + 1) != GetNumRows() || (size_t)(colLast + 1) != GetNumCols())
///		fprintf(stderr, " [%ld:%ld, %ld:%ld]", (long) rowFirst, (long) rowLast, (long) colFirst, (long) colLast);
///	fprintf(stderr, " ######\n\n");

///	if (IsEmpty())
///	{
///		fprintf(stderr, "(empty)\n");
///		return;
///	}

///	PrintRange rowRange(rowFirst, rowLast, GetNumRows());
///	PrintRange colRange(colFirst, colLast, GetNumCols());

///	if (rowRange.IsEmpty() || colRange.IsEmpty())
///	{
///		fprintf(stderr, "(empty)\n");
///		return;
///	}

///	const auto& us = *this;
///	if (rowRange.begin > 0)
///		fprintf(stderr, "...\n");
///	for (size_t i = rowRange.begin; i < rowRange.end; i++)
///	{
///		if (i == rowRange.skipBegin)        // insert ... between the two blocks if any
///		{
///			fprintf(stderr, "...\n");
///			i = rowRange.skipEnd;
///		}
///		if (colRange.begin > 0)             // ... at line start
///			fprintf(stderr, "...\t");
///		for (size_t j = colRange.begin; j < colRange.end; j++)
///		{
///			if (j == colRange.skipBegin)
///			{
///				fprintf(stderr, "...\t");
///				j = colRange.skipEnd;
///			}
///			fprintf(stderr, "%.10f\t", (double)us(i, j));
///		}
///		if (colRange.end < GetNumCols())    // ... at line end
///			fprintf(stderr, "..");
///		fprintf(stderr, "\n");
///	}
///	if (rowRange.end < GetNumRows())
///		fprintf(stderr, "...\n");
///}

///template <class ElemType>
///void CPUMatrix<ElemType>::Print(const char* matrixName /*=nullptr*/) const
///{
///	Print(matrixName, 0, GetNumRows() - 1, 0, GetNumCols() - 1);
///}

//// file I/O
////matrixName is used to verify that correct matrix is read.
///template <class ElemType>
///void CPUMatrix<ElemType>::ReadFromFile(FILE*, const char* /*matrixName*/)
///{
///	RuntimeError("not implemented");
///}

////matrixName is used to verify that correct matrix is read.
///template <class ElemType>
///void CPUMatrix<ElemType>::WriteToFile(FILE*, const char* /*matrixName*/)
///{
///	RuntimeError("not implemented");
///}

////assume each column is an input sample. Each sample is stored in [channel, row, col]  (r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11)
///template <class ElemType>
///CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignPackedConvolutionInput(const CPUMatrix<ElemType>& inputSubBatch,
///																		const size_t inputWidth, size_t inputHeight, size_t inputChannels,
///																		const size_t outputWidth, size_t outputHeight, size_t /*outputChannels*/,
///																		const size_t kernelWidth, size_t kernelHeight, size_t horizontalSubsample, size_t verticalSubsample,
///																		bool zeroPadding)
///{
///	if (verticalSubsample > kernelHeight || horizontalSubsample > kernelWidth)
///		LogicError("Arguments verticalSubsample (or horitzontalSubsample) must be less or equal than kernelHeight (or kernelWidth)");

///	const size_t packedInputRows = kernelWidth * kernelHeight * inputChannels;
///	const size_t packedInputColsPerSample = outputWidth * outputHeight; // output size per channel
///	const size_t inputDim = inputWidth * inputHeight * inputChannels;
///	const size_t smallBatchSize = inputSubBatch.GetNumCols();
///	const long inputHeightTimesChannel = (long) (inputHeight * inputChannels);
///	Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
///	if (zeroPadding)
///		SetValue((ElemType) 0);

///	const long halfKernelWidth = (long) kernelWidth / 2;
///	const long halfKernelHeight = (long) kernelHeight / 2;

///#pragma omp parallel for // each input element is copied to many places
///	for (long sample = 0; sample < smallBatchSize; sample++)
///	{
///		for (long id = 0; id < inputDim; id++)
///		{
///			// IN_ELEM_ROWPOS(channel, row, col) = (channel + (row + col * inputHeight) * inputChannels)
///			// IN_ELEM_COLPOS = sample

///			const long y = id / inputHeightTimesChannel;   // inputCol
///			const long nXC = id % inputHeightTimesChannel; // channel + inputRow*inputChannels
///			const long x = nXC / (long) inputChannels;     // inputRow
///			const long c = nXC % (long) inputChannels;     // channel

///			long x0 = 0, y0 = 0, x1 = 0, y1 = 0;
///			if (zeroPadding)
///			{
///				x0 = (long) max((ElemType)0, ceil((x - (ElemType)kernelHeight + 1.0f + halfKernelHeight) / (ElemType)verticalSubsample)); // row : first wrow in which x is in
///				x1 = (long) (x + halfKernelHeight - x0 * verticalSubsample);                                                      // first posxInKernel
///				y0 = (long) max((ElemType)0, ceil((y - (ElemType)kernelWidth + 1.0f + halfKernelWidth) / (ElemType)horizontalSubsample)); // col : first wcol in which y is in
///				y1 = (long) (y + halfKernelWidth - y0 * horizontalSubsample);                                                     // first posyInKernel
///			}
///			else
///			{
///				x0 = (long) max((ElemType)0, ceil((x - (ElemType)kernelHeight + 1) / (ElemType)verticalSubsample));  // row : first wrow in which x is in
///				x1 = (long) (x - x0 * verticalSubsample);                                                    // first posxInKernel
///				y0 = (long) max((ElemType)0, ceil((y - (ElemType)kernelWidth + 1) / (ElemType)horizontalSubsample)); // col : first wcol in which y is in
///				y1 = (long) (y - y0 * horizontalSubsample);                                                  // first posyInKernel
///			}

///			assert(x1 >= 0 && x1 < kernelHeight && y1 >= 0 && y1 < kernelWidth);

///			// PACK_ELEM_ROWPOS(channel, posxInKernel, posyInKernel) = (channel * kernelWidth * kernelHeight + posxInKernel + posyInKernel * kernelHeight)
///			// PACK_ELEM_COLPOS(sample, wrow, wcol) = (sample*packedInputColsPerSample + outputHeight*wcol + wrow

///			ElemType currentInputValue = inputSubBatch(id, sample);
///			long packColBase = (long) (sample * packedInputColsPerSample + y0 * outputHeight);
///			for (long wcol = y0, posyInKernel = y1; wcol < (long) outputWidth && posyInKernel >= 0; wcol++, posyInKernel -= (long) horizontalSubsample)
///			{
///				long packRowBase = (long) (c * kernelWidth * kernelHeight + posyInKernel * kernelHeight);
///				for (long wrow = x0, posxInKernel = x1; wrow < (long) outputHeight && posxInKernel >= 0; wrow++, posxInKernel -= (long) verticalSubsample)
///				{
///					const long packRow = packRowBase + posxInKernel;
///					const long packCol = packColBase + wrow;
///					(*this)(packRow, packCol) = currentInputValue;
///				}
///				packColBase += (long) outputHeight;
///			}
///		}
///	}

///	return *this;
///}
////assume each column is an input sample. Each sample is stored in [channel, row, col]  (r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11)
///template <class ElemType>
///CPUMatrix<ElemType>& CPUMatrix<ElemType>::UnpackConvolutionInput(CPUMatrix<ElemType>& inputSubBatch,
///																	const size_t inputWidth, size_t inputHeight, size_t inputChannels,
///																	const size_t outputWidth, size_t outputHeight, size_t /*outputChannels*/,
///																	const size_t kernelWidth, size_t kernelHeight, size_t horizontalSubsample, size_t verticalSubsample,
///																	bool zeroPadding) const
///{
///	if (verticalSubsample > kernelHeight || horizontalSubsample > kernelWidth)
///		LogicError("Arguments verticalSubsample (or horizonSubsample) must be less than or equal to kernelHeight (or kernelWidth)");

///	const size_t packedInputColsPerSample = outputWidth * outputHeight; // output size per channel
///	const size_t inputDim = inputWidth * inputHeight * inputChannels;
///	const size_t smallBatchSize = inputSubBatch.GetNumCols();
///	const long inputHeightTimesChannel = (long) (inputHeight * inputChannels);

///	const long halfKernelWidth = (long) kernelWidth / 2;
///	const long halfKernelHeight = (long) kernelHeight / 2;

///#pragma omp parallel for // each input element is copied to many places
///	for (long sample = 0; sample < smallBatchSize; sample++)
///	{
///		for (long id = 0; id < inputDim; id++)
///		{
///			// IN_ELEM_ROWPOS(channel, row, col) = (channel + (row + col * inputHeight) * inputChannels)
///			// IN_ELEM_COLPOS = sample

///			const long y = id / inputHeightTimesChannel;   // inputCol
///			const long nXC = id % inputHeightTimesChannel; // channel + inputRow*inputChannels
///			const long x = nXC / (long) inputChannels;     // inputRow
///			const long c = nXC % (long) inputChannels;     // channel

///			long x0 = 0, y0 = 0, x1 = 0, y1 = 0;
///			if (zeroPadding)
///			{
///				x0 = (long) max((ElemType)0, ceil((x - (ElemType) kernelHeight + 1.0f + halfKernelHeight) / (ElemType) verticalSubsample)); // row : first wrow in which x is in
///				x1 = (long) (x + halfKernelHeight - x0 * verticalSubsample);                                                      // first posxInKernel
///				y0 = (long) max((ElemType)0, ceil((y - (ElemType) kernelWidth + 1.0f + halfKernelWidth) / (ElemType) horizontalSubsample)); // col : first wcol in which y is in
///				y1 = (long) (y + halfKernelWidth - y0 * horizontalSubsample);                                                     // first posyInKernel
///			}
///			else
///			{
///				x0 = (long) max((ElemType)0, ceil((x - (ElemType) kernelHeight + 1) / (ElemType) verticalSubsample));  // row : first wrow in which x is in
///				x1 = (long) (x - x0 * verticalSubsample);                                                    // first posxInKernel
///				y0 = (long) max((ElemType)0, ceil((y - (ElemType) kernelWidth + 1) / (ElemType) horizontalSubsample)); // col : first wcol in which y is in
///				y1 = (long) (y - y0 * horizontalSubsample);                                                  // first posyInKernel
///			}

///			assert(x1 >= 0 && x1 < kernelHeight && y1 >= 0 && y1 < kernelWidth);

///			// PACK_ELEM_ROWPOS(channel, posxInKernel, posyInKernel) = (channel * kernelWidth * kernelHeight + posxInKernel + posyInKernel * kernelHeight)
///			// PACK_ELEM_COLPOS(sample, wrow, wcol) = (sample*packedInputColsPerSample + outputHeight*wcol + wrow

///			ElemType currentInputValue = inputSubBatch(id, sample);
///			long packColBase = (long) (sample * packedInputColsPerSample + y0 * outputHeight);
///			for (long wcol = y0, posyInKernel = y1; wcol < (long) outputWidth && posyInKernel >= 0; wcol++, posyInKernel -= (long) horizontalSubsample)
///			{
///				long packRowBase = (long) (c * kernelWidth * kernelHeight + posyInKernel * kernelHeight);
///				for (long wrow = x0, posxInKernel = x1; wrow < (long) outputHeight && posxInKernel >= 0; wrow++, posxInKernel -= (long) verticalSubsample)
///				{
///					const long packRow = packRowBase + posxInKernel;
///					const long packCol = packColBase + wrow;
///					currentInputValue += (*this)(packRow, packCol);
///				}
///				packColBase += (long) outputHeight;
///			}
///			inputSubBatch(id, sample) = currentInputValue;
///		}
///	}

///	return inputSubBatch;
///}

////assume each column is an input sample. Each sample is stored in  (r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11)
///template <class ElemType>
///CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignMaxPoolingResult(const CPUMatrix<ElemType>& inputBatch, size_t channels,
///																	const size_t /*inputWidth*/, size_t inputHeight, size_t /*inputSizePerSample*/,
///																	const size_t /*outputWidth*/, size_t outputHeight, size_t outputSizePerSample,
///																	const size_t windowWidth, size_t windowHeight, size_t horizontalSubsample, size_t verticalSubsample)
///{
///	const long inputHeightTimesChannel = (long) (inputHeight * channels);
///	const long outputHeightTimesChannel = (long) (outputHeight * channels);
///	const size_t batchSize = inputBatch.GetNumCols();
///	Resize(outputSizePerSample, batchSize);

//// IN_ELEM_ROWPOS(channel, row, col) = (channel + (row + col * inputHeight) * channels)
//// IN_ELEM_COLPOS = sample

//// OUT_ELEM_ROWPOS(channel, wrow, wcol) = (channel + (wrow + wcol * outputHeight) * channels)
//// OUT_ELEM_COLPOS = sample

///#pragma omp parallel for
///	for (long sample = 0; sample < (long) batchSize; sample++)
///	{
///		for (long outputIndexWithinSample = 0; outputIndexWithinSample < outputSizePerSample; outputIndexWithinSample++)
///		{
///			const long y = outputIndexWithinSample / outputHeightTimesChannel;   // wcol
///			const long nXC = outputIndexWithinSample % outputHeightTimesChannel; // channel + wrow*channels
///			const long x = (long) (nXC / channels);                              // wrow
///			const long c = (long) (nXC % channels);                              // channel

///			ElemType maxVal = -FLT_MAX;
///			ElemType minVal = FLT_MAX;
///			const long rowInWindowBase = (long) ((x * verticalSubsample + y * horizontalSubsample * inputHeight) * channels + c);
///			for (long colInWindow = 0; colInWindow < windowWidth; colInWindow++)
///			{
///				long rowInInput = rowInWindowBase + colInWindow * inputHeightTimesChannel;
///				for (long rowInWindow = 0; rowInWindow < windowHeight; rowInWindow++)
///				{
///					const ElemType val = inputBatch(rowInInput, sample); // pf[rowInWindow*channels];
///					maxVal = std::max(maxVal, val);
///					minVal = std::min(minVal, val);
///					rowInInput += (long) channels;
///				}
///			}

///			(*this)(outputIndexWithinSample, sample) = maxVal;
///		}
///	}

///	return *this;
///}

///template <class ElemType>
///CPUMatrix<ElemType>& CPUMatrix<ElemType>::AddMaxPoolingGradient(const CPUMatrix<ElemType>& outputGradientBatch, const CPUMatrix<ElemType>& inputBatch, const CPUMatrix<ElemType>& outputBatch,
///																const size_t channels,
///																const size_t /*inputWidth*/, size_t inputHeight, size_t inputSizePerSample,
///																const size_t outputWidth, size_t outputHeight, size_t /*outputSizePerSample*/,
///																const size_t windowWidth, size_t windowHeight, size_t horizontalSubsample, size_t verticalSubsample)
///{
///	size_t batchSize = inputBatch.GetNumCols();
///	const long inputHeightTimesChannel = (long) (inputHeight * channels);
///	const long outputHeightTimesChannel = (long) (outputHeight * channels);

//// IN_ELEM_ROWPOS(channel, row, col) = (channel + (row + col * inputHeight) * channels)
//// IN_ELEM_COLPOS = sample

//// OUT_ELEM_ROWPOS(channel, wrow, wcol) = (channel + (wrow + wcol * outputHeight) * channels)
//// OUT_ELEM_COLPOS = sample

///#pragma omp parallel for
///	for (long sample = 0; sample < batchSize; sample++)
///	{
///		for (long inputIndexWithinSample = 0; inputIndexWithinSample < inputSizePerSample; inputIndexWithinSample++)
///		{
///			const long y = inputIndexWithinSample / inputHeightTimesChannel;   // col in input
///			const long nXC = inputIndexWithinSample % inputHeightTimesChannel; // channel + row*chanels
///			const long x = (long) (nXC / channels);                            // row in input
///			const long c = (long) (nXC % channels);                            // channel

///			long startOutX = (long) max((ElemType)0, ceil((x - (ElemType) windowHeight + 1) / (ElemType) verticalSubsample));          // inclusive start
///			long endOutX = (long) ((x / verticalSubsample < outputHeight - 1) ? x / verticalSubsample : outputHeight - 1);   // inclusive end
///			long startOutY = (long) max((ElemType)0, ceil((y - (ElemType) windowWidth + 1) / (ElemType) horizontalSubsample));         // inclusive start
///			long endOutY = (long) ((y / horizontalSubsample < outputWidth - 1) ? y / horizontalSubsample : outputWidth - 1); // inclusive end

///			ElemType inputValue = inputBatch(inputIndexWithinSample, sample);
///			for (long outY = startOutY; outY <= endOutY; outY++)
///			{
///				for (long outX = startOutX; outX <= endOutX; outX++)
///				{
///					long outputIndex = (long) (outY * outputHeightTimesChannel + outX * channels + c);
///					if (inputValue == outputBatch(outputIndex, sample))
///						(*this)(inputIndexWithinSample, sample) += outputGradientBatch(outputIndex, sample);
///				}
///			}
///		}
///	}

///	return *this;
///}
///template <class ElemType>
///CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignAveragePoolingResult(const CPUMatrix<ElemType>& inputBatch, size_t channels,
///																		const size_t /*inputWidth*/, size_t inputHeight, size_t /*inputSizePerSample*/,
///																		const size_t /*outputWidth*/, size_t outputHeight, size_t outputSizePerSample,
///																		const size_t windowWidth, size_t windowHeight, size_t horizontalSubsample, size_t verticalSubsample)
///{
///	const long inputHeightTimesChannel = (long) (inputHeight * channels);
///	const long outputHeightTimesChannel = (long) (outputHeight * channels);
///	const size_t batchSize = inputBatch.GetNumCols();
///	const size_t windowSize = windowWidth * windowHeight;
///	Resize(outputSizePerSample, batchSize);

//// IN_ELEM_ROWPOS(channel, row, col) = (channel + (row + col * inputHeight) * channels)
//// IN_ELEM_COLPOS = sample

//// OUT_ELEM_ROWPOS(channel, wrow, wcol) = (channel + (wrow + wcol * outputHeight) * channels)
//// OUT_ELEM_COLPOS = sample

///#pragma omp parallel for
///	for (long sample = 0; sample < batchSize; sample++)
///	{
///		for (long outputIndexWithinSample = 0; outputIndexWithinSample < outputSizePerSample; outputIndexWithinSample++)
///		{
///			const long y = outputIndexWithinSample / outputHeightTimesChannel;   // wcol
///			const long nXC = outputIndexWithinSample % outputHeightTimesChannel; // channel + wrow*channels
///			const long x = (long) (nXC / channels);                              // wrow
///			const long c = (long) (nXC % channels);                              // channel

///			ElemType sum = 0;
///			const long rowInWindowBase = (long) ((x * verticalSubsample + y * horizontalSubsample * inputHeight) * channels + c);
///			for (long colInWindow = 0; colInWindow < windowWidth; colInWindow++)
///			{
///				long rowInInput = rowInWindowBase + colInWindow * inputHeightTimesChannel;
///				for (long rowInWindow = 0; rowInWindow < windowHeight; rowInWindow++)
///				{
///					sum += inputBatch(rowInInput, sample);
///					rowInInput += (long) channels;
///				}
///			}

///			(*this)(outputIndexWithinSample, sample) = sum / windowSize;
///		}
///	}

///	return *this;
///}

///template <class ElemType>
///CPUMatrix<ElemType>& CPUMatrix<ElemType>::AddAveragePoolingGradient(const CPUMatrix<ElemType>& outputGradientBatch,
///																	const size_t channels,
///																	const size_t /*inputWidth*/, size_t inputHeight, size_t inputSizePerSample,
///																	const size_t outputWidth, size_t outputHeight, size_t /*outputSizePerSample*/,
///																	const size_t windowWidth, size_t windowHeight, size_t horizontalSubsample, size_t verticalSubsample)
///{
///	size_t batchSize = outputGradientBatch.GetNumCols();
///	const long inputHeightTimesChannel = (long) (inputHeight * channels);
///	const long outputHeightTimesChannel = (long) (outputHeight * channels);
///	const long windowSize = (long) (windowWidth * windowHeight);

//// IN_ELEM_ROWPOS(channel, row, col) = (channel + (row + col * inputHeight) * channels)
//// IN_ELEM_COLPOS = sample

//// OUT_ELEM_ROWPOS(channel, wrow, wcol) = (channel + (wrow + wcol * outputHeight) * channels)
//// OUT_ELEM_COLPOS = sample

///#pragma omp parallel for
///	for (long sample = 0; sample < batchSize; sample++)
///	{
///		for (long inputIndexWithinSample = 0; inputIndexWithinSample < inputSizePerSample; inputIndexWithinSample++)
///		{
///			const long y = inputIndexWithinSample / inputHeightTimesChannel;   // col in input
///			const long nXC = inputIndexWithinSample % inputHeightTimesChannel; // channel + row*chanels
///			const long x = nXC / (long) channels;                              // row in input
///			const long c = nXC % (long) channels;                              // channel

///			long startOutX = (long) max((ElemType)0, ceil((x - (ElemType) windowHeight + 1) / (ElemType) verticalSubsample));               // inclusive start
///			long endOutX = (long) ((x / verticalSubsample < outputHeight - 1) ? x / (long) verticalSubsample : outputHeight - 1); // inclusive end
///			long startOutY = (long) max((ElemType)0, ceil((y - (ElemType) windowWidth + 1) / (ElemType) horizontalSubsample));              // inclusive start
///			long endOutY = (long) ((y / horizontalSubsample < outputWidth - 1) ? y / horizontalSubsample : outputWidth - 1);      // inclusive end

///			for (long outY = startOutY; outY <= endOutY; outY++)
///			{
///				for (long outX = startOutX; outX <= endOutX; outX++)
///				{
///					long outputIndex = outY * outputHeightTimesChannel + outX * (long) channels + c;
///					(*this)(inputIndexWithinSample, sample) += outputGradientBatch(outputIndex, sample) / windowSize;
///				}
///			}
///		}
///	}

///	return *this;
///}

//// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

///template <class ElemType>
///void CPUMatrix<ElemType>::ConvolutionForward(const CPUMatrix<ElemType>& kernel, const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIwht,
///												const CPUMatrix<int>& mpRowRun, const CPUMatrix<int>& runs, CPUMatrix<ElemType>& output) const
///{
///#pragma omp parallel for
///	for (int64_t sample = 0; sample < (int64_t)output.GetNumCols(); sample++)
///	{
///		for (size_t row = 0; row < output.GetNumRows(); row++)
///		{
///			int colBase = mpRowCol(row, 0);
///			int ivBase = mpRowIwht(row, 0);
///			assert(0 <= colBase && colBase < GetNumRows());

///			ElemType sum = 0;
///			int i0 = mpRowRun(row, 0);
///			int skip = runs(i0++, 0);
///			int size = runs(i0++, 0);
///			int imask = i0 + size;
///			for (int i = 0; i < size; i++)
///			{
///				if (runs(imask + i, 0) == 0)
///					continue;
///				int dcol = runs(i0 + i, 0);
///				assert(0 <= colBase + dcol && colBase + dcol < GetNumRows());
///				sum += kernel.GetData()[ivBase + skip + i] * (*this)(colBase + dcol, sample);
///			}
///			output(row, sample) = sum;
///		}
///	}
///}

///template <class ElemType>
///void CPUMatrix<ElemType>::ConvolutionBackwardData(const CPUMatrix<ElemType>& kernel, const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIwht,
///													const CPUMatrix<int>& mpRowRun, const CPUMatrix<int>& runs, CPUMatrix<ElemType>& grad) const
///{
///#pragma omp parallel for
///	for (int64_t sample = 0; sample < (int64_t)GetNumCols(); sample++)
///	{
///		for (size_t row = 0; row < GetNumRows(); row++)
///		{
///			int colBase = mpRowCol(row, 0);
///			int ivBase = mpRowIwht(row, 0);
///			assert(0 <= colBase && colBase < grad.GetNumRows());

///			ElemType curGrad = (*this)(row, sample);

///			int i0 = mpRowRun(row, 0);
///			int skip = runs(i0++, 0);
///			int size = runs(i0++, 0);
///			int imask = i0 + size;
///			for (int i = 0; i < size; i++)
///			{
///				if (runs(imask + i, 0) == 0)
///					continue;
///				int dcol = runs(i0 + i, 0);
///				assert(0 <= colBase + dcol && colBase + dcol < grad.GetNumRows());
///				grad(colBase + dcol, sample) += curGrad * kernel.GetData()[ivBase + skip + i];
///			}
///		}
///	}
///}

///template <class ElemType>
///void CPUMatrix<ElemType>::ConvolutionBackwardKernel(const CPUMatrix<ElemType>& in, const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIwht,
///													const CPUMatrix<int>& mpRowRun, const CPUMatrix<int>& runs, CPUMatrix<ElemType>& kernelGrad) const
///{
///	// Do NOT parallelize these loops!
///	for (size_t sample = 0; sample < GetNumCols(); sample++)
///	{
///		for (size_t row = 0; row < GetNumRows(); row++)
///		{
///			int colBase = mpRowCol(row, 0);
///			int ivBase = mpRowIwht(row, 0);
///			assert(0 <= colBase && colBase < in.GetNumRows());

///			ElemType curGrad = (*this)(row, sample);

///			int i0 = mpRowRun(row, 0);
///			int skip = runs(i0++, 0);
///			int size = runs(i0++, 0);
///			int imask = i0 + size;
///			for (int i = 0; i < size; i++)
///			{
///				if (runs(imask + i, 0) == 0)
///					continue;
///				int dcol = runs(i0 + i, 0);
///				assert(0 <= colBase + dcol && colBase + dcol < in.GetNumRows());
///				kernelGrad.GetData()[ivBase + skip + i] += curGrad * in(colBase + dcol, sample);
///			}
///		}
///	}
///}

///template <class ElemType>
///void CPUMatrix<ElemType>::UnrollConvolutionInput(size_t unrollCols, size_t mapOutSize, const CPUMatrix<int>& mpRowCol,
///													const CPUMatrix<int>& mpRowRun, const CPUMatrix<int>& runs, CPUMatrix<ElemType>& output) const
///{
///	size_t batchSize = GetNumCols();

///#pragma omp parallel for
///	for (int64_t sample = 0; sample < (int64_t)batchSize; sample++)
///	{
///		for (size_t row = 0; row < mapOutSize; row++)
///		{
///			int colBase = mpRowCol(row, 0);
///			assert(0 <= colBase && colBase < GetNumRows());

///			int i0 = mpRowRun(row, 0);
///			int skip = runs(i0++, 0);
///			int size = runs(i0++, 0);
///			int imask = i0 + size;
///			for (int i = 0; i < size; i++)
///			{
///				if (runs(imask + i, 0) == 0)
///					continue;
///				int dcol = runs(i0 + i, 0);
///				assert(0 <= colBase + dcol && colBase + dcol < GetNumRows());
///				output.GetData()[(row * batchSize + sample) * unrollCols + skip + i] = (*this)(colBase + dcol, sample);
///			}
///		}
///	}
///}

///template <class ElemType>
///void CPUMatrix<ElemType>::UnrollConvolutionOutput(size_t unrollCols, size_t mapInCount, size_t mapOutCount, const CPUMatrix<int>& mpRowCol,
///													const CPUMatrix<int>& mpRowRun, const CPUMatrix<int>& runs, CPUMatrix<ElemType>& output) const
///{
///	if (mpRowCol.GetNumRows() % mapOutCount != 0)
///		InvalidArgument("The number of rows in mpRowCol must be multiple of mapOutCount");
///	size_t mapOutSize = mpRowCol.GetNumRows() / mapOutCount;
///	size_t batchSize = GetNumCols();

///	size_t kernelSize = runs(1, 0);
///	if (kernelSize % mapInCount != 0)
///		InvalidArgument("kernelSize must be multiple of mapInCount");
///	size_t kernelMapSize = kernelSize / mapInCount;

///#pragma omp parallel for
///	for (int64_t sample = 0; sample < (int64_t)GetNumCols(); sample++)
///	{
///		for (size_t row = 0; row < mapOutSize; row++)
///		{
///			int colBase = mpRowCol(row, 0);

///			int i0 = mpRowRun(row, 0);
///			int skip = runs(i0++, 0);
///			int size = runs(i0++, 0);
///			int imask = i0 + size;
///			for (int i = 0; i < std::min(size, (int)kernelMapSize); i++)
///			{
///				if (runs(imask + i, 0) == 0)
///					continue;
///				int dcol = runs(i0 + i, 0);
///				size_t isrc = row;
///				size_t idst = ((colBase + dcol) * batchSize + sample) * unrollCols + ((skip + i) % kernelMapSize) * mapOutCount;
///				for (size_t outMap = 0; outMap < mapOutCount; outMap++, isrc += mapOutSize)
///				{
///					assert(isrc < GetNumElements());
///					assert(idst + outMap < output.GetNumElements());

///					output.GetData()[idst + outMap] = (*this)(isrc, sample);
///				}
///			}
///		}
///	}
///}

///template <class ElemType>
///void CPUMatrix<ElemType>::UnrollConvolutionInputForKernelBackprop(size_t mapOutSize, const CPUMatrix<int>& mpRowCol,
///																	const CPUMatrix<int>& mpRowRun, const CPUMatrix<int>& runs, CPUMatrix<ElemType>& output) const
///{
///	size_t batchSize = GetNumCols();
///	size_t unrollCols = mapOutSize * batchSize;

///#pragma omp parallel for
///	for (int64_t sample = 0; sample < (int64_t)batchSize; sample++)
///	{
///		for (size_t row = 0; row < mapOutSize; row++)
///		{
///			int colBase = mpRowCol(row, 0);
///			assert(0 <= colBase && colBase < GetNumRows());

///			int i0 = mpRowRun(row, 0);
///			int skip = runs(i0++, 0);
///			int size = runs(i0++, 0);
///			int imask = i0 + size;
///			for (int i = 0; i < size; i++)
///			{
///				if (runs(imask + i, 0) == 0)
///					continue;
///				int dcol = runs(i0 + i, 0);
///				assert(0 <= colBase + dcol && colBase + dcol < GetNumRows());
///				size_t idst = (skip + i) * unrollCols + row * batchSize + sample;
///				assert(idst < output.GetNumElements());
///				output.GetData()[idst] = (*this)(colBase + dcol, sample);
///			}
///		}
///	}
///}

///template <class ElemType>
///void CPUMatrix<ElemType>::MaxPoolingForward(const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIndices, const CPUMatrix<int>& indices, CPUMatrix<ElemType>& output) const
///{
///#pragma omp parallel for
///	for (int64_t sample = 0; sample < (int64_t)output.GetNumCols(); sample++)
///	{
///		for (size_t row = 0; row < output.GetNumRows(); row++)
///		{
///			int colBase = mpRowCol(row, 0);
///			assert(0 <= colBase && colBase < GetNumRows());

///			assert(std::numeric_limits<ElemType>::has_infinity);
///			ElemType res = -std::numeric_limits<ElemType>::infinity();

///			int i0 = mpRowIndices(row, 0);
///			int size = indices(i0++, 0);
///			assert(size > 0);
///			for (int i = 0; i < size; i++)
///			{
///				int dcol = indices(i0 + i, 0);
///				assert(0 <= colBase + dcol && colBase + dcol < GetNumRows());
///				res = std::max(res, (*this)(colBase + dcol, sample));
///			}
///			output(row, sample) = res;
///		}
///	}
///}

///template <class ElemType>
///void CPUMatrix<ElemType>::MaxPoolingBackward(const CPUMatrix<ElemType>& out, const CPUMatrix<ElemType>& in,
///												const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIndices, const CPUMatrix<int>& indices,
///												CPUMatrix<ElemType>& grad, bool accumulateGradient) const
///{
///	if (!accumulateGradient)
///		grad.SetValue((ElemType)0);

///#pragma omp parallel for
///	for (int64_t sample = 0; sample < (int64_t)GetNumCols(); sample++)
///	{
///		for (size_t row = 0; row < GetNumRows(); row++)
///		{
///			int colBase = mpRowCol(row, 0);
///			assert(0 <= colBase && colBase < grad.GetNumRows());

///			int i0 = mpRowIndices(row, 0);
///			int size = indices(i0++, 0);
///			assert(size > 0);
///			ElemType g = (*this)(row, sample);
///			ElemType m = out(row, sample);
///			for (int i = 0; i < size; i++)
///			{
///				int dcol = indices(i0 + i, 0);
///				assert(0 <= colBase + dcol && colBase + dcol < grad.GetNumRows());
///				if (in(colBase + dcol, sample) >= m)
///				{
///#pragma omp atomic
///					grad(colBase + dcol, sample) += g;
///					break;
///				}
///			}
///		}
///	}
///}

//// For each image, for each ROI, this function treats that ROI as an image
//// and does max pooling so that it has output size pooledHeight x pooledWidth.
//// It loops over each location in the output tensor, computes which ROI
//// and image should populate that location, computes the subset of the image
//// corresponding to the ROI and which pixels in that subset should go into the
//// output location, then takes the max value over that window.
//// src: Images              [W x H x C x N]
//// roiData: ROIs            [4 x numROIs x N],
//// dst: Pooled ROIs         [PW x PH x C x numROIs x N]
//// argmax: max positions    [PW x PH x C x numROIs x N]
//// spatialScale             ratio of input feature map to the original image.
//// where PW = Pooled Width, PH = Pooled Height, C = Channels, N = Batch Size
///template <class ElemType>
///void CPUMatrix<ElemType>::MaxROIPoolingForward(size_t numRois, size_t numImg, size_t channels, size_t width, size_t height,
///												const size_t pooledWidth, size_t pooledHeight, const CPUMatrix<ElemType>& roiData, CPUMatrix<ElemType>& output,
///												CPUMatrix<ElemType>& argmax, double spatialScale) const
///{
///	size_t roiOutputSize = pooledHeight * pooledWidth * channels;

///#pragma omp parallel for
///	for (int imgIdx = 0; imgIdx < numImg; imgIdx++)
///	{
///		auto img = GetColumnSlice(imgIdx, 1);
///		auto rois = roiData.GetColumnSlice(imgIdx, 1);
///#pragma omp parallel for
///		for (int roiIdx = 0; roiIdx < numRois; roiIdx++)
///		{
///			// each ROI is 4 elements: (x, y, w, h).
///			int base = roiIdx * 4;

///			// roi points represent the absolute location of the roi
///			// in the original image.
///			ElemType scX1 = rois(base, (ElemType)0);
///			ElemType scY1 = rois(base + (ElemType)1, (ElemType)0);
///			ElemType scX2 = rois(base + (ElemType)2, (ElemType)0);
///			ElemType scY2 = rois(base + (ElemType)3, (ElemType)0);

///			// compute actual spatial location of the ROI in our featuremap.
///			size_t x1 = (size_t)round(scX1 * spatialScale);
///			size_t y1 = (size_t)round(scY1 * spatialScale);
///			size_t x2 = (size_t)round(scX2 * spatialScale);
///			size_t y2 = (size_t)round(scY2 * spatialScale);

///			ElemType roiW = (ElemType)max(x2 - x1+1, (size_t)1);
///			ElemType roiH = (ElemType)max(y2 - y1+1, (size_t)1);

///			const ElemType winW = roiW / (ElemType)pooledWidth;
///			const ElemType winH = roiH / (ElemType)pooledHeight;

///			// inspired by Ross Girshick fast-rcnn caffe cpu: https://github.com/rbgirshick/fast-rcnn
///			// loop over spatial locations in output.
///#pragma omp parallel for
///			for (int outw = 0; outw < pooledWidth; outw++)
///			{
///				for (int outh = 0; outh < pooledHeight; outh++)
///				{
///					// compute the top left corner of the input
///					// spatial window corresponding to this output unit
///					size_t hstart = (size_t)floor(outh * winH);
///					size_t wstart = (size_t)floor(outw * winW);

///					// compute bottom right corner (not included)
///					size_t hend = (size_t)ceil((outh + 1) * winH);
///					size_t wend = (size_t)ceil((outw + 1) * winW);

///					// offset window based on ROI top left corner.
///					// these indices are into the input slice.
///					hstart = min(max(hstart + y1, (size_t)0), height);
///					wstart = min(max(wstart + x1, (size_t)0), width);
///					hend = min(max(hend + y1, (size_t)0), height);
///					wend = min(max(wend + x1, (size_t)0), width);

///					bool isempty = (hend <= hstart) || (wend <= wstart);

///					for (size_t c = 0; c < channels; c++)
///					{
///						// [W x H x C x R x N]; R = ROIs per image
///						size_t outputIdx = roiIdx * roiOutputSize + outw + outh * pooledWidth + c * pooledHeight * pooledWidth;
///						size_t maxidx = 0;
///						ElemType maxval = isempty ? (ElemType)0 : (ElemType)-FLT_MAX;
///						size_t baseIdx = c * height * width;

///						for (size_t h = hstart; h < hend; h++)
///						{
///							for (size_t w = wstart; w < wend; w++)
///							{
///								// stored argmax indices are relative to the current channel.
///								size_t dataIdx = w + h * width;
///								if (img(baseIdx + dataIdx, 0) > maxval)
///								{
///									maxval = img(baseIdx + dataIdx, 0);
///									maxidx = dataIdx;
///								}
///							}
///						}
///						output(outputIdx, imgIdx) = maxval;
///						argmax(outputIdx, imgIdx) = maxidx;
///					}
///				}
///			}
///		}
///	}
///}

//// This function loops over locations in the input to the ROIPoolingNode (image locations).
//// It loops over the ROIs corresponding to that image, seeing which ones could contain the current location
//// in their output. For each ROI, it checks the argmax data to see if that ROI indeed chose
//// this pixel location as the maximum. If so, it increments the gradient term for the input location.
///template <class ElemType>
///void CPUMatrix<ElemType>::MaxROIPoolingBackward(size_t numRois, size_t numImg, size_t channels, size_t width, size_t height,
///												const size_t pooledWidth, size_t pooledHeight, const CPUMatrix<ElemType>& roiData, CPUMatrix<ElemType>& grad,
///												CPUMatrix<ElemType>& argmax, double spatialScale) const
///{
///	// loop over images in the batch.
///#pragma omp parallel for
///	for (int imgIdx = 0; imgIdx < numImg; imgIdx++)
///	{
///		// ROIs for this image. length 4*numRois;
///		auto rois = roiData.GetColumnSlice(imgIdx, 1).GetData();
///		// gradient values for all ROIs from this image. length numRois*pooledHeight*pooledWidth*channels;
///		auto pooledGrad = GetColumnSlice(imgIdx, 1).GetData();
///		auto argmaxCol = argmax.GetColumnSlice(imgIdx, 1).GetData();

///		// loop over spatial locations in the image.
///#pragma omp parallel for
///		for (int w = 0; w < width; w++)
///		{
///#pragma omp parallel for
///			for (int h = 0; h < width; h++)
///			{
///				// loop over the ROIs seeing which ones contain this location.
///				for (int roiN = 0; roiN < numRois; roiN++)
///				{
///					// each ROI is 4 elements: (x, y, w, h).
///					int roiOffset = roiN * 4;

///					// ROI data points represent the absolute location of the roi
///					// in the original image.
///					size_t roiStartW = (size_t)round(rois[roiOffset + 0] * spatialScale);
///					size_t roiStartH = (size_t)round(rois[roiOffset + 1] * spatialScale);
///					size_t roiEndW = (size_t)round(rois[roiOffset + 2] * spatialScale);
///					size_t roiEndH = (size_t)round(rois[roiOffset + 3] * spatialScale);

///					size_t roiWidth = max(roiEndW - roiStartW+1, (size_t)1);
///					size_t roiHeight = max(roiEndH - roiStartH+1, (size_t)1);

///					// skip this ROI if it doesn't contain the current input location.
///					bool inROI = (w >= roiStartW && w < roiStartW + roiWidth &&
///						h >= roiStartH && h < roiStartH + roiHeight);
///					if (!inROI)
///						continue;

///					ElemType winH = (ElemType)roiHeight / (ElemType)pooledHeight;
///					ElemType winW = (ElemType)roiWidth / (ElemType)pooledWidth;

///					// what pooled nodes in the output for this ROI could have pooled this input location?
///					size_t phstart = (size_t)((h - roiStartH) / winH);
///					size_t pwstart = (size_t)((w - roiStartW) / winW);
///					size_t phend = (size_t)(ceil((h - roiStartH + 1) / winH));
///					size_t pwend = (size_t)(ceil((w - roiStartW + 1) / winW));

///					phstart = min(max(phstart, (size_t)0), pooledHeight);
///					phend = min(max(phend, (size_t)0), pooledHeight);
///					pwstart = min(max(pwstart, (size_t)0), pooledWidth);
///					pwend = min(max(pwend, (size_t)0), pooledWidth);

///					for (size_t c = 0; c < channels; c++)
///					{
///						ElemType gradient = 0;
///						// [W x H x C x N]
///						size_t index = w + h*width + c*height*width;
///						// go right up to channel c of the current ROI.
///						size_t offset = (roiN * channels + c) * pooledWidth * pooledHeight;
///						const ElemType* offsetPoolGrad = pooledGrad + offset;
///						const ElemType* offsetArgmax = argmaxCol + offset;
///						for (size_t ph = phstart; ph < phend; ph++)
///						{
///							for (size_t pw = pwstart; pw < pwend; pw++)
///							{
///								if ((size_t)offsetArgmax[ph * pooledWidth + pw] == (w + h * width))
///								{
///									gradient += offsetPoolGrad[ph * pooledWidth + pw];
///								}
///							}
///						}

///#pragma omp atomic
///						grad(index, imgIdx) += gradient;
///					}
///				}
///			}
///		}
///	}
///}

///template <class ElemType>
///void CPUMatrix<ElemType>::MaxUnpooling(const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIndices,
///										const CPUMatrix<int>& indices, const CPUMatrix<ElemType>& poolInput,
///										CPUMatrix<ElemType>& input) const
///{
///#pragma omp parallel for
///	for (int64_t sample = 0; sample < (int64_t)GetNumCols(); sample++)
///	{
///		for (size_t row = 0; row < GetNumRows(); row++)
///		{
///			int colBase = mpRowCol(row, 0);
///			assert(0 <= colBase && colBase < input.GetNumRows());

///			int i0 = mpRowIndices(row, 0);
///			int size = indices(i0++, 0);
///			assert(size > 0);

///			ElemType curMax = poolInput(colBase + indices(i0, 0), sample);
///			ElemType prevMax = curMax;
///			int imax = 0;
///			for (int i = 1; i < size; i++)
///			{
///				int dcol = indices(i0 + i, 0);
///				assert(0 <= colBase + dcol && colBase + dcol < poolInput.GetNumRows());
///				curMax = std::max(curMax, poolInput(colBase + dcol, sample));
///				if (curMax > prevMax)
///				{
///					prevMax = curMax;
///					imax = i;
///				}
///			}

///			int dcol = indices(i0 + imax, 0);
///			assert(0 <= colBase + dcol && colBase + dcol < input.GetNumRows());
///			input(colBase + dcol, sample) = (*this)(row, sample);

///			//int i = (int)poolIn(row, sample);
///			//assert(0 <= i && i < size);
///			//int dcol = indices(i0 + i, 0);
///			//assert(0 <= colBase + dcol && colBase + dcol < input.GetNumRows());
///			//input(colBase + dcol, sample) = (*this)(row, sample);
///		}
///	}
///}

///template <class ElemType>
///void CPUMatrix<ElemType>::AveragePoolingForward(const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIndices, const CPUMatrix<int>& indices, CPUMatrix<ElemType>& output, bool poolIncludePad) const
///{
///#pragma omp parallel for
///	for (int64_t sample = 0; sample < (int64_t)output.GetNumCols(); sample++)
///	{
///		for (size_t row = 0; row < output.GetNumRows(); row++)
///		{
///			int colBase = mpRowCol(row, 0);
///			assert(0 <= colBase && colBase < GetNumRows());

///			ElemType sum = 0;

///			int i0 = mpRowIndices(row, 0);
///			int size = indices(i0++, 0);
///			assert(size > 0);
///			for (int i = 0; i < size; i++)
///			{
///				int dcol = indices(i0 + i, 0);
///				assert(0 <= colBase + dcol && colBase + dcol < GetNumRows());
///				sum += (*this)(colBase + dcol, sample);
///			}
///			// Note that we divide by size which is the number of actual elements (does not include padding).
///			// if poolIncludePad == true, use avg_pool_include_pad
///			if (poolIncludePad)
///				size = indices(0, 0);
///			output(row, sample) = sum / size;
///		}
///	}
///}

///template <class ElemType>
///void CPUMatrix<ElemType>::AveragePoolingBackward(const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIndices, const CPUMatrix<int>& indices, CPUMatrix<ElemType>& grad, bool poolIncludePad, bool accumulateGradient) const
///{
///	if (!accumulateGradient)
///		grad.SetValue((ElemType)0);

///#pragma omp parallel for
///	for (int64_t sample = 0; sample < (int64_t)GetNumCols(); sample++)
///	{
///		for (size_t row = 0; row < GetNumRows(); row++)
///		{
///			int colBase = mpRowCol(row, 0);
///			assert(0 <= colBase && colBase < grad.GetNumRows());

///			int i0 = mpRowIndices(row, 0);
///			int size = indices(i0++, 0);
///			int tmp = size;
///			if (poolIncludePad)
///				size = indices(0, 0);
///			assert(size > 0);
///			ElemType g = (*this)(row, sample) / size;
///			size = tmp;
///			for (int i = 0; i < size; i++)
///			{
///				int dcol = indices(i0 + i, 0);
///				assert(0 <= colBase + dcol && colBase + dcol < grad.GetNumRows());
///#pragma omp atomic
///				grad(colBase + dcol, sample) += g;
///			}
///		}
///	}
///}

///template <class ElemType>
///template <class StatType>
///void CPUMatrix<ElemType>::BatchNormalizationForward(const CPUMatrix<StatType>& scale, const CPUMatrix<StatType>& bias, bool inferenceOnly, double expAvgFactor, double blendFactor,
///													CPUMatrix<StatType>& runMean, CPUMatrix<StatType>& runVariance, CPUMatrix<ElemType>& out, double epsilon,
///													CPUMatrix<StatType>& saveMean, CPUMatrix<StatType>& saveInvStdDev) const
///{
///	if (GetNumRows() % scale.GetNumRows() != 0)
///		LogicError("The number of rows of this matrx must be multiple of the number of rows of the scale matrix");

///	if (!inferenceOnly || expAvgFactor != 0 || blendFactor != 1)
///		RuntimeError("Batch normalization training on CPU is not yet implemented");

///	saveMean.Resize(0, 0); // only doing inference: these two are not produced
///	saveInvStdDev.Resize(0, 0);

///	bool spatial = GetNumRows() != scale.GetNumRows();
///	if (spatial)
///	{
///		size_t spatialSize = GetNumRows() / scale.GetNumRows();
///#pragma omp parallel for
///		for (long icol = 0; icol < out.GetNumCols(); icol++)
///		{
///			for (long irow = 0; irow < out.GetNumRows(); irow++)
///			{
///				size_t imap = irow / spatialSize;
///				ElemType stdDev = sqrt(runVariance(imap, 0) + epsilon);
///				out(irow, icol) = (ElemType)(scale(imap, 0) * ((*this)(irow, icol) - runMean(imap, 0)) / stdDev + bias(imap, 0));
///			}
///		}
///	}
///	else
///	{
///#pragma omp parallel for
///		for (long icol = 0; icol < out.GetNumCols(); icol++)
///		{
///			for (long irow = 0; irow < out.GetNumRows(); irow++)
///			{
///				ElemType stdDev = sqrt(runVariance(irow, 0) + epsilon);
///				out(irow, icol) = (ElemType)(scale(irow, 0) * ((*this)(irow, icol) - runMean(irow, 0)) / stdDev + bias(irow, 0));
///			}
///		}
///	}
///}

///template <class ElemType>
///template <class StatType>
///void CPUMatrix<ElemType>::BatchNormalizationBackward(const CPUMatrix<ElemType>& in, CPUMatrix<ElemType>& grad, const CPUMatrix<StatType>& scale, double blendFactor,
///														const CPUMatrix<StatType>& saveMean, const CPUMatrix<StatType>& saveInvStdDev,
///														CPUMatrix<StatType>& scaleGrad, CPUMatrix<StatType>& biasGrad) const
///{
///	UNUSED(in); UNUSED(grad); UNUSED(scale); UNUSED(blendFactor), UNUSED(saveMean); UNUSED(saveInvStdDev); UNUSED(scaleGrad); UNUSED(biasGrad);
///	RuntimeError("Batch normalization training on CPU is not yet implemented");
///}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//			Static BLAS Functions
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

//	Matrix-matrix multiply with col-major matrices (a and b are not transposed)
//		c =  a * b

template <class ElemType>
void CPUMatrix<ElemType>::Multiply(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c)
{
	MultiplyAndWeightedAdd(1.0, a, false, b, false, 0.0, c);
}

//	Matrix-matrix multiply with col-major matrices (a and b may be transposed)
//		c =  op(a) * op(b)

template <class ElemType>
void CPUMatrix<ElemType>::Multiply(const CPUMatrix<ElemType>& a, bool transposeA, const CPUMatrix<ElemType>& b, bool transposeB, CPUMatrix<ElemType>& c)
{
	MultiplyAndWeightedAdd(1.0, a, transposeA, b, transposeB, 0.0, c);
}

//	Matrix-matrix multiply with col-major matrices (a and b may be transposed)
//		c =  op(a) * op(b) + c

template <class ElemType>
void CPUMatrix<ElemType>::MultiplyAndAdd(const CPUMatrix<ElemType>& a, bool transposeA, const CPUMatrix<ElemType>& b, bool transposeB, CPUMatrix<ElemType>& c)
{
	MultiplyAndWeightedAdd(1.0, a, transposeA, b, transposeB, 1.0, c);
}

// Matrix-matrix multiply with col-major matrices (a and b may be transposed)
//		c = alpha * op(a) * op(b) + beta * c

template <class ElemType>
void CPUMatrix<ElemType>::MultiplyAndWeightedAdd(ElemType alpha,
												const CPUMatrix<ElemType>& a, bool transposeA,
												const CPUMatrix<ElemType>& b, bool transposeB,
												ElemType beta, CPUMatrix<ElemType>& c)
{
	if (a.IsEmpty() || b.IsEmpty()) return;

	int m, n, k, l;
	int lda, ldb, ldc;
	CBLAS_TRANSPOSE mklTransA;
	CBLAS_TRANSPOSE mklTransB;

	if (transposeA)
	{
		m = (int)a.GetNumCols();
		k = (int)a.GetNumRows();
		lda = k;
		mklTransA = CBLAS_TRANSPOSE::CblasTrans;
	}
	else
	{
		m = (int)a.GetNumRows();
		k = (int)a.GetNumCols();
		lda = m;
		mklTransA = CBLAS_TRANSPOSE::CblasNoTrans;
	}

	if (transposeB)
	{
		l = (int)b.GetNumCols();
		n = (int)b.GetNumRows();
		ldb = n;
		mklTransB = CBLAS_TRANSPOSE::CblasTrans;
	}
	else
	{
		l = (int)b.GetNumRows();
		n = (int)b.GetNumCols();
		ldb = l;
		mklTransB = CBLAS_TRANSPOSE::CblasNoTrans;
	}

	if (k != l) InvalidArgument("MultiplyAndWeightedAdd; The inner dimensions of A and B must match");

	if (beta == 0) c.Resize(m, n);
	else c.VerifySize(m, n, "MultiplyAndWeightedAdd");

	ldc = (int)c.GetNumRows();
	if (std::is_same<ElemType, double>::value)
	{
		cblas_dgemm((CBLAS_ORDER) (int)MatrixOrder::ColMajor, mklTransA, mklTransB, m, n, k, alpha, reinterpret_cast<double*>(a.GetData()), lda, 
														reinterpret_cast<double*>(b.GetData()), ldb, beta, reinterpret_cast<double*>(c.GetData()), ldc);
	}
	else if (std::is_same<ElemType, float>::value)
	{
#pragma warning(suppress : 4244)
		cblas_sgemm((CBLAS_ORDER) (int)MatrixOrder::ColMajor, mklTransA, mklTransB, m, n, k, alpha, reinterpret_cast<float*>(a.GetData()), lda,
														reinterpret_cast<float*>(b.GetData()), ldb, beta, reinterpret_cast<float*>(c.GetData()), ldc);
	}
	else
	{
		RuntimeError("MultiplyAndWeightedAdd; Unsupported data type");
	}
}

//template <class ElemType>
//void CPUMatrix<ElemType>::MultiplyAndWeightedAdd(ElemType alpha,
//												const CPUMatrix<ElemType>& a, bool transposeA,
//												const CPUMatrix<ElemType>& b, bool transposeB,
//												ElemType beta, CPUMatrix<ElemType>& c,
//												shared_ptr<QuantizedMultiplier<ElemType>> pQuantizedMultiplier)
//{
//	if (a.IsEmpty() || b.IsEmpty()) return;
//
//	int m, n, k, l;
//	int lda, ldb, ldc;
//	CBLAS_TRANSPOSE mklTransA;
//	CBLAS_TRANSPOSE mklTransB;
//
//	if (transposeA)
//	{
//		m = (int)a.GetNumCols();
//		k = (int)a.GetNumRows();
//		lda = k;
//		mklTransA = CBLAS_TRANSPOSE::CblasTrans;
//	}
//	else
//	{
//		m = (int)a.GetNumRows();
//		k = (int)a.GetNumCols();
//		lda = m;
//		mklTransA = CBLAS_TRANSPOSE::CblasNoTrans;
//	}
//
//	if (transposeB)
//	{
//		l = (int)b.GetNumCols();
//		n = (int)b.GetNumRows();
//		ldb = n;
//		mklTransB = CBLAS_TRANSPOSE::CblasTrans;
//	}
//	else
//	{
//		l = (int)b.GetNumRows();
//		n = (int)b.GetNumCols();
//		ldb = l;
//		mklTransB = CBLAS_TRANSPOSE::CblasNoTrans;
//	}
//
//	if (k != l) InvalidArgument("MultiplyAndWeightedAdd; The inner dimensions of A and B must match");
//
//	if (beta == 0) c.Resize(m, n);
//	else c.VerifySize(m, n, "MultiplyAndWeightedAdd");
//
//	ldc = (int)c.GetNumRows();
//	if (pQuantizedMultiplier == nullptr)
//	{
//		if (std::is_same<ElemType, double>::value)
//		{
//			cblas_dgemm((CBLAS_ORDER) (int)MatrixOrder::ColMajor, mklTransA, mklTransB, m, n, k, alpha, reinterpret_cast<double*>(a.GetData()), lda, 
//															reinterpret_cast<double*>(b.GetData()), ldb, beta, reinterpret_cast<double*>(c.GetData()), ldc);
//		}
//		else if (std::is_same<ElemType, float>::value)
//		{
//#pragma warning(suppress : 4244)
//			cblas_sgemm((CBLAS_ORDER) (int)MatrixOrder::ColMajor, mklTransA, mklTransB, m, n, k, alpha, reinterpret_cast<float*>(a.GetData()), lda,
//															reinterpret_cast<float*>(b.GetData()), ldb, beta, reinterpret_cast<float*>(c.GetData()), ldc);
//		}
//		else
//		{
//			RuntimeError("MultiplyAndWeightedAdd; Unsupported data type");
//		}
//	}
//	else
//	{
//		if (mklTransA == CBLAS_TRANSPOSE::CblasTrans || mklTransB == CBLAS_TRANSPOSE::CblasTrans)
//			LogicError("MultiplyAndWeightedAdd; Quantized multiplier doesn't support transpose");
//
//		pQuantizedMultiplier->Multiply(m, n, k, a.GetData(), b.GetData(), c.GetData());
//	}
//}

//	c = alpha * a * v + beta * c

template <class ElemType>
void CPUMatrix<ElemType>::ColumnwiseScaleAndWeightedAdd(ElemType alpha, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& v, ElemType beta, CPUMatrix<ElemType>& c)
{
	if (v.GetNumRows() != 1 && v.GetNumCols() != 1)
		InvalidArgument("ColumnwiseScaleAndWeightedAdd; The argument V must be a vector");
	if (v.GetNumElements()!=a.GetNumCols())
		InvalidArgument("ColumnwiseScaleAndWeightedAdd; V size=%lu is not same as A columns=%lu", v.GetNumElements(), a.GetNumCols());

	const ElemType* vd = v.GetData();
	if (beta==0)
	{
		c.Resize(a.GetNumRows(), a.GetNumCols());
#pragma omp parallel for
		foreach_coord(i, j, c)
			c(i, j) = alpha * a(i, j) * vd[j];
	}
	else
	{
		c.VerifySize(a.GetNumRows(), a.GetNumCols(), "ColumnwiseScaleAndWeightedAdd");
#pragma omp parallel for
		foreach_coord(i, j, c)
			c(i, j) = alpha * a(i, j) * vd[j] + beta * c(i, j);
	}
}

template <class ElemType>
void CPUMatrix<ElemType>::Multiply1x1AndWeightedAdd(ElemType alpha, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, ElemType beta, CPUMatrix<ElemType>& c)
{
	if (a.GetNumElements() != 1)
		InvalidArgument("Multiply1x1AndWeightedAdd; The argument a must be a scalar");

	ElemType f = alpha * a.GetFirstItem();
	if (beta == 0) // don't even read the memory if beta is 0
#pragma omp parallel for
		foreach_coord (i, j, c)
			c(i, j) = b(i, j) * f;
	else
#pragma omp parallel for
		foreach_coord (i, j, c)
			c(i, j) = b(i, j) * f + c(i, j) * beta;
}

//	Matrix-scalar multiply with col-major matrices
//		a = alpha * a
template <class ElemType>
void CPUMatrix<ElemType>::Scale(ElemType alpha, CPUMatrix<ElemType>& a)
{
	if (a.IsEmpty()) LogicError("Scale; Input matrix A is empty");

	const int m = (int)a.GetNumRows();
	const int n = (int)a.GetNumCols();
	const int len = m * n;
	const int incx = 1;

	if (alpha == 0)
	{
		memset(a.GetData(), 0, len*sizeof(ElemType));
	}
	else if (std::is_same<ElemType, double>::value)
	{
		cblas_dscal(len, alpha, reinterpret_cast<double*>(a.GetData()), incx);
	}
	else if (std::is_same<ElemType, float>::value)
	{
#pragma warning(suppress : 4244)
		cblas_sscal(len, alpha, reinterpret_cast<float*>(a.GetData()), incx);
	}
	else if (std::is_same<ElemType, half>::value)
	{
		ElemType* p = a.GetData();
		for (int j=0; j<len; ++j) *p++ *= alpha;
	}
	else RuntimeError("Scale; Unsupported data type");
}

//	Matrix multiply with col-major matrices
//		a = alpha[1,1] * a

//template <class ElemType>
//void CPUMatrix<ElemType>::Scale(CPUMatrix<ElemType> alpha, CPUMatrix<ElemType>& a)
//{
//	if (a.IsEmpty())
//		LogicError("Scale:  Input matrix a is empty");
//	if (alpha.GetNumElements() != 1)
//		LogicError("Matrix alpha must be 1x1");
//	CPUMatrix<ElemType>::Scale(alpha(0, 0), a);
//}

//	Matrix-scalar multiply with col-major matrices
//		c = alpha * a

template <class ElemType>
void CPUMatrix<ElemType>::Scale(ElemType alpha, const CPUMatrix<ElemType>& a, CPUMatrix<ElemType>& c)
{
	if (a.IsEmpty()) LogicError("Scale; The matrix A is empty");

	c.Resize(a.GetNumRows(), a.GetNumCols());
	long m = (long) c.GetNumElements();
	if (alpha == 0) { memset(c.GetData(), 0, m*sizeof(ElemType)); return; }

	long m4 = m & ~3;
	ElemType* pa = a.GetData();
	ElemType* pc = c.GetData();
#pragma omp parallel for
	for (long i = 0; i < m4; i += 4)
	{
		pc[i]   = alpha * pa[i];
		pc[i+1] = alpha * pa[i+1];
		pc[i+2] = alpha * pa[i+2];
		pc[i+3] = alpha * pa[i+3];
	}
	for (long i = m4; i<m; i++) pc[i] = alpha * pa[i];
}

//	Matrix-scalar multiply with col-major matrices
//		c = alpha * a + c
//
//	if a is a column vector, add to all columns of c
//	if a is a row vector, add to all rows of c
//	if a is a scalar, add to all rows of c

template <class ElemType>
void CPUMatrix<ElemType>::ScaleAndAdd(ElemType alpha, const CPUMatrix<ElemType>& a, CPUMatrix<ElemType>& c)
{
	if (a.IsEmpty() || c.IsEmpty())
		LogicError("ScaleAndAdd; The matrix A or C is empty");

	if (a.GetNumRows() != 1 && a.GetNumCols() != 1) // A is not a col or row vector
	{
		const int m = (int)a.GetNumRows();
		const int n = (int)a.GetNumCols();
		const int len = m * n;
		const int incx = 1;
		const int incy = 1;

		if (c.GetNumRows() != m || c.GetNumCols() != n)
			InvalidArgument("ScaleAndAdd; C[%lu,%lu] doesn't match A[%d,%d]", c.GetNumRows(), c.GetNumCols(), m, n);

		if (std::is_same<ElemType, double>::value)
		{
			cblas_daxpy(len, alpha, reinterpret_cast<double*>(a.GetData()), incx, reinterpret_cast<double*>(c.GetData()), incy);
		}
		else if (std::is_same<ElemType, float>::value)
		{
#pragma warning(suppress : 4244)
			cblas_saxpy(len, alpha, reinterpret_cast<float*>(a.GetData()), incx, reinterpret_cast<float*>(c.GetData()), incy);
		}
		else if (std::is_same<ElemType, half>::value)
		{
			ElemType* pa = a.GetData();
			ElemType* pc = c.GetData();
			for (int j=0; j<len; ++j) *pc++ += alpha*(*pa++);
		}
		else RuntimeError("ScaleAndAdd; Unsupported data type");
	}
	else if (a.GetNumElements() == 1) // scalar, add to all elements
	{
		ElemType v = alpha * a(0, 0);
		long m = (long) c.GetNumRows(), n = (long) c.GetNumCols();
		long m4 = m & ~3;
#pragma omp parallel for
		for (long j = 0; j < n; j++)
		{
			ElemType* pc = c.GetDataCol(j);
			for (long i = 0; i < m4; i += 4)
			{
				pc[i] += v;
				pc[i+1] += v;
				pc[i+2] += v;
				pc[i+3] += v;
			}
			for (long i = m4; i < m; i++) pc[i] += v;
		}
	}
	else if (a.GetNumCols() == 1) // col vector, add it to all columns
	{
		int m = (int)c.GetNumRows();
		if (a.GetNumRows()!=m) InvalidArgument("ScaleAndAdd; C[%lu,] doesn't match A[%lu,]", c.GetNumRows(), a.GetNumRows());

		ElemType* aBufPtr = a.GetData();
		ElemType* cBufPtr = c.GetData();
		if (std::is_same<ElemType, double>::value)
		{
#pragma omp parallel for
			foreach_column (j, c)
			{
				cblas_daxpy(m, alpha, reinterpret_cast<double*>(aBufPtr), 1, reinterpret_cast<double*>(cBufPtr + c.ColumnPos(j)), 1);
			}
		}
		else if (std::is_same<ElemType, float>::value)
		{
#pragma omp parallel for
			foreach_column (j, c)
			{
#pragma warning(suppress : 4244)
				cblas_saxpy(m, alpha, reinterpret_cast<float*>(aBufPtr), 1, reinterpret_cast<float*>(cBufPtr + c.ColumnPos(j)), 1);
			}
		}
		else RuntimeError("ScaleAndAdd; Unsupported data type");
	}
	else // row vector, add it to all rows
	{
		int m = (int)c.GetNumRows();
		int n = (int)c.GetNumCols();
		if (n != (int)a.GetNumCols()) InvalidArgument("ScaleAndAdd; C[,%lu] doesn't match A[,%lu]", c.GetNumCols(), a.GetNumCols());

		ElemType* aBufPtr = a.GetData();
		ElemType* cBufPtr = c.GetData();
		if (std::is_same<ElemType, double>::value)
		{
#pragma omp parallel for
			foreach_row (i, c)
			{
				cblas_daxpy(n, alpha, reinterpret_cast<double*>(aBufPtr), 1, reinterpret_cast<double*>(cBufPtr + i), m);
			}
		}
		else if (std::is_same<ElemType, float>::value)
		{
#pragma omp parallel for
			foreach_row (i, c)
			{
#pragma warning(suppress : 4244)
				cblas_saxpy(n, alpha, reinterpret_cast<float*>(aBufPtr), 1, reinterpret_cast<float*>(cBufPtr + i), m);
			}
		}
		else RuntimeError("ScaleAndAdd; Unsupported data type");
	}
}

//	c = alpha * (a-b) + c
template <class ElemType>
void CPUMatrix<ElemType>::AddScaledDifference(ElemType alpha, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c)
{
	if (a.IsEmpty()) LogicError("AssignScaledDifference; Empty matrix A[%lu,%lu]", a.GetNumRows(), a.GetNumCols());
	if (a.GetNumRows()!=b.GetNumRows() || a.GetNumRows()!=c.GetNumRows() ||
		a.GetNumCols()!=b.GetNumCols() || a.GetNumCols()!=c.GetNumCols())
		InvalidArgument("AddScaledDifference; A[%lu,%lu], B[%lu,%lu], C[%lu,%lu] must have same size",
						a.GetNumRows(), a.GetNumCols(), b.GetNumRows(), b.GetNumCols(), c.GetNumRows(), c.GetNumCols());

	ElemType* aBufPtr = a.GetData();
	ElemType* bBufPtr = b.GetData();
	ElemType* cBufPtr = c.GetData();
	long m = (long) c.GetNumElements();
	long m4 = m & ~3;
#pragma omp parallel for
	// four-way unrolling
	for (long i = 0; i < m4; i += 4)
	{
		cBufPtr[i] += alpha * (aBufPtr[i] - bBufPtr[i]);
		cBufPtr[i+1] += alpha * (aBufPtr[i+1] - bBufPtr[i+1]);
		cBufPtr[i+2] += alpha * (aBufPtr[i+2] - bBufPtr[i+2]);
		cBufPtr[i+3] += alpha * (aBufPtr[i+3] - bBufPtr[i+3]);
	}
	// handle remaining stuffs
	for (long i = m4; i < m; i++)
	{
		cBufPtr[i] += alpha * (aBufPtr[i] - bBufPtr[i]);
	}
}

//	c = alpha * (a-b)
template <class ElemType>
void CPUMatrix<ElemType>::AssignScaledDifference(ElemType alpha, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c)
{
	if (a.IsEmpty()) LogicError("AssignScaledDifference; Empty matrix A[%lu,%lu]", a.GetNumRows(), a.GetNumCols());
	if (a.GetNumRows()!=b.GetNumRows() || a.GetNumCols()!=b.GetNumCols())
		InvalidArgument("AssignScaledDifference; A[%lu,%lu], B[%lu,%lu] must have same size",
						a.GetNumRows(), a.GetNumCols(), b.GetNumRows(), b.GetNumCols());

	if (&c != &a && &c != &b) c.Resize(a.GetNumRows(), a.GetNumCols());

	ElemType* aBufPtr = a.GetData();
	ElemType* bBufPtr = b.GetData();
	ElemType* cBufPtr = c.GetData();
	long m = (long) c.GetNumElements();
	long m4 = m & ~3;
#pragma omp parallel for
	// four-way unrolling
	for (long i = 0; i < m4; i += 4)
	{
		cBufPtr[i] = alpha * (aBufPtr[i] - bBufPtr[i]);
		cBufPtr[i+1] = alpha * (aBufPtr[i+1] - bBufPtr[i+1]);
		cBufPtr[i+2] = alpha * (aBufPtr[i+2] - bBufPtr[i+2]);
		cBufPtr[i+3] = alpha * (aBufPtr[i+3] - bBufPtr[i+3]);
	}
	// handle remaining stuffs
	for (long i = m4; i < m; i++)
	{
		cBufPtr[i] = alpha * (aBufPtr[i] - bBufPtr[i]);
	}
}

template <class ElemType>
void CPUMatrix<ElemType>::InnerProduct(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c, bool isColWise)
{
	if (a.IsEmpty() || b.IsEmpty())
		LogicError("InnerProduct:  one of the input matrices is empty");

	const int m = (int)a.GetNumRows();
	const int n = (int)a.GetNumCols();
	const int k = (int)b.GetNumRows();
	const int l = (int)b.GetNumCols();

	assert(m > 0 && n > 0 && k > 0 && l > 0); // converting from size_t to int may cause overflow
	if (m != k || n != l)
		InvalidArgument("InnerProduct: Matrices a and b should have same dimension");

	if ((isColWise && m == 1) || !isColWise && n == 1) // in this case it's equivalent to element-wise product
	{
		c.AssignElementProductOf(a, b);
	}
	else if (isColWise) // col-wise
	{
		c.Resize(1, n);

		ElemType* aBufPtr = a.GetData();
		ElemType* bBufPtr = b.GetData();
		if (std::is_same<ElemType, double>::value)
		{
#pragma omp parallel for
			foreach_column (j, c)
			{
				c(0, j) = (ElemType) cblas_ddot(m, reinterpret_cast<double*>(aBufPtr + a.ColumnPos(j)), 1, reinterpret_cast<double*>(bBufPtr + b.ColumnPos(j)), 1);
			}
		}
		else if (std::is_same<ElemType, float>::value)
		{
#pragma omp parallel for
			foreach_column (j, c)
			{
#pragma warning(suppress : 4244)
				c(0, j) = (ElemType) cblas_sdot(m, reinterpret_cast<float*>(aBufPtr + a.ColumnPos(j)), 1, reinterpret_cast<float*>(bBufPtr + b.ColumnPos(j)), 1);
			}
		}
		else
		{
			RuntimeError("InnerProduct; Unsupported data type");
		}
	}
	else
	{
		c.Resize(m, 1);

		ElemType* aBufPtr = a.GetData();
		ElemType* bBufPtr = b.GetData();
		if (std::is_same<ElemType, double>::value)
		{
#pragma omp parallel for
			foreach_row (i, c)
			{
				c(i, 0) = cblas_ddot(n, reinterpret_cast<double*>(aBufPtr + i), m, reinterpret_cast<double*>(bBufPtr + i), m);
			}
		}
		else if (std::is_same<ElemType, float>::value)
		{
#pragma omp parallel for
			foreach_row (i, c)
			{
#pragma warning(suppress : 4244)
				c(i, 0) = cblas_sdot(n, reinterpret_cast<float*>(aBufPtr + i), m, reinterpret_cast<float*>(bBufPtr + i), m);
			}
		}
		else
		{
			RuntimeError("InnerProduct; Unsupported data type");
		}
	}
}

//	Compute singular value decomposition as A = U*SIGMA*VT
//	W is used as temp working memory
///template <class ElemType>
///void CPUMatrix<ElemType>::SVD(const CPUMatrix<ElemType>& A, CPUMatrix<ElemType>& SIGMA, CPUMatrix<ElemType>& U, CPUMatrix<ElemType>& VT, CPUMatrix<ElemType>& W)
///{
///	if (A.IsEmpty())
///		LogicError("SVD:  input matrix is empty");

///	int info;
///	int m, n, lda, ldu, ldvt;
///	m = (int)A.GetNumRows();
///	n = (int)A.GetNumCols();
///	W.GetNumRows(); // W is used as temp working memory
///	lda = m;
///	ldu = m;
///	ldvt = n;
///	U.Resize(m, m);
///	SIGMA.Resize(std::min(m, n), 1);
///	VT.Resize(n, n);

///#if CNTK_UWP
///	RuntimeError("Error, LAPACKE_*gesvd is not supported for UWP.\n");
///#else
///	if (std::is_same<ElemType, double>::value)
///	{
///		std::vector<double> superb(std::max(std::min(m, n) - 1, 1));
///		info = LAPACKE_dgesvd((int)MatrixOrder::ColMajor, 'A', 'A', (int)m, (int)n, reinterpret_cast<double*>(A.GetData()), (int)lda, reinterpret_cast<double*>(SIGMA.GetData()),
///			reinterpret_cast<double*>(U.GetData()), (int)ldu, reinterpret_cast<double*>(VT.GetData()), (int)ldvt, &superb[0]);
///	}
///	else if (std::is_same<ElemType, float>::value)
///	{
///		std::vector<float> superb(std::max(std::min(m, n) - 1, 1));
///		info = LAPACKE_sgesvd((int)MatrixOrder::ColMajor, 'A', 'A', (int)m, (int)n, reinterpret_cast<float*>(A.GetData()), (int)lda, reinterpret_cast<float*>(SIGMA.GetData()),
///			reinterpret_cast<float*>(U.GetData()), (int)ldu, reinterpret_cast<float*>(VT.GetData()), (int)ldvt, &superb[0]);
///	}
///	else
///	{
///		RuntimeError("SVD; Unsupported data type");
///	}
///#endif

///	if (info > 0)
///	{
///		RuntimeError("The algorithm computing SVD failed to converge.\n");
///	}
///}

///template <class ElemType>
///void CPUMatrix<ElemType>::AssignSoftmaxSum(const CPUMatrix<ElemType>& softmax, CPUMatrix<ElemType>& c)
///{
///	ElemType log_likelihood = 0.0;
///	size_t batch_size = GetNumCols();
///#pragma omp parallel for reduction(+ : log_likelihood)
///	for (int instance_id = 0; instance_id < batch_size; instance_id++)
///	{
///		int sample = (int)(*this)(0, instance_id);
///		log_likelihood += softmax(instance_id, sample);
///	}
///	c(0, 0) = -log_likelihood;
///}

///template <class ElemType>
///void CPUMatrix<ElemType>::AssignNCEUnnormalizedEval(const CPUMatrix<ElemType>& a,
///													const CPUMatrix<ElemType>& b, const CPUMatrix<ElemType>& bias, CPUMatrix<ElemType>& c)
////this: samples+probs
//// a:   hidden
//// b:   embedding
//// tmp:  softmax
////  c: loglikelihood
///{
///	ElemType log_likelihood = 0.0;
///	size_t batch_size = GetNumCols();
///#pragma omp parallel for reduction(+ : log_likelihood)
///	for (int instance_id = 0; instance_id < batch_size; instance_id++)
///	{
///		int sample = -(int)(*this)(0, instance_id);
///		ElemType score = bias(sample, 0);
///		for (int dim = 0; dim < b.GetNumRows(); dim++)
///			score += b(dim, sample) * a(dim, instance_id);
///		log_likelihood += score;
///	}
///	c(0, 0) = -log_likelihood;
///}

////samples+prob                         gradient           hidden               embedding          embedding/hidden
////a.m_CPUMatrix->AssignNCEDerivative(*tmp.m_CPUMatrix, *a.m_CPUMatrix, *b.m_CPUMatrix, inputIndex, *c.m_CPUMatrix);
///template <class ElemType>
///CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignNCEDerivative(const CPUMatrix<ElemType>& tmp, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, size_t inputIndex, CPUMatrix<ElemType>& c)
///{
///	size_t sample_size = GetNumRows() / 2;
///	size_t batch_size = GetNumCols();
///	if (inputIndex == 1)
///	{
///#pragma omp parallel for
///		for (int instance_id = 0; instance_id < batch_size; instance_id++)
///			for (int sample_id = 0; sample_id < sample_size; sample_id++)
///			{
///				int sample = (int)(*this)(2 * sample_id, instance_id);
///				for (int dim = 0; dim < b.GetNumRows(); dim++)
///					c(dim, instance_id) -= b(dim, sample) * tmp(sample_id, instance_id);
///			}
///	}
///	else if (inputIndex == 2)
///	{
///		int i_blocks = omp_get_num_threads() * 16;
//// Assume only one block in k direction.
//// We don't need to explicitly block in the j direction.
///#pragma omp parallel for
///		for (int ib = 0; ib < i_blocks; ib++)
///			for (int instance_id = 0; instance_id < batch_size; instance_id++)
///				for (int sample_id = 0; sample_id < sample_size; sample_id++)
///				{
///					int sample = (int)(*this)(2 * sample_id, instance_id);
///					if (sample % i_blocks == ib)
///						for (int dim = 0; dim < b.GetNumRows(); dim++)
///							c(dim, sample) -= a(dim, instance_id) * tmp(sample_id, instance_id);
///				}
///	}
///	else if (inputIndex == 3)
///	{
///		// Assume only one block in k direction.
///		// We don't need to explicitly block in the j direction.
///		for (int instance_id = 0; instance_id < batch_size; instance_id++)
///			for (int sample_id = 0; sample_id < sample_size; sample_id++)
///			{
///				int sample = (int)(*this)(2 * sample_id, instance_id);
///				c(0, sample) -= tmp(sample_id, instance_id);
///			}
///	}
///	else
///		InvalidArgument("The argument inputIndex must be 1 or 2 or 3");
///	return *this;
///}

///template <class ElemType>
///void CPUMatrix<ElemType>::AssignNoiseContrastiveEstimation(const CPUMatrix<ElemType>& a,
///															const CPUMatrix<ElemType>& b, const CPUMatrix<ElemType>& bias, CPUMatrix<ElemType>& tmp, CPUMatrix<ElemType>& c)
////this: samples+probs
//// a:   hidden
//// b:   embedding
//// tmp:  softmax
//// c: loglikelihood
///{
///	double log_likelihood = 0.0;
///	size_t sample_size = GetNumRows() / 2;
///	size_t batch_size = GetNumCols();
///	size_t num_noise_samples = sample_size - 1;
///	double log_num_noise_samples = std::log(num_noise_samples);
///#pragma omp parallel for reduction(+ : log_likelihood)
///	for (int instance_id = 0; instance_id < batch_size; instance_id++)
///		for (int sample_id = 0; sample_id < sample_size; sample_id++)
///		{
///			int sample = (int)(*this)(2 * sample_id, instance_id);
///			double score = bias(0, sample);
///			for (int dim = 0; dim < b.GetNumRows(); dim++)
///				score += (double)(a(dim, instance_id) * b(dim, sample));
///			double sample_prob = -(*this)(2 * sample_id+1, instance_id);
///			if (sample_id == 0)
///				sample_prob = -sample_prob;
///			double score_noise = log_num_noise_samples + sample_prob;
///			double z = LogAdd(score, score_noise);
///			double logprob = score - z;
///			double logprob_noise = score_noise - z;
///			tmp(sample_id, instance_id) = (ElemType) -std::exp(logprob);
///			if (sample_id == 0)
///				tmp(sample_id, instance_id) += 1;
///			log_likelihood += sample_id == 0 ? logprob : logprob_noise;
///		}
///	c(0, 0) = (ElemType) -log_likelihood;
///}


//// c[ci,cj] += a[ai,aj]
///template <class ElemType>
///void CPUMatrix<ElemType>::AddElementToElement(ElemType beta, const CPUMatrix<ElemType>& a, size_t ai, size_t aj, CPUMatrix<ElemType>& c, size_t ci, size_t cj)
///{
///	if (ai >= a.GetNumRows() || aj >= a.GetNumCols() ||
///		ci >= c.GetNumRows() || cj >= c.GetNumCols())
///		InvalidArgument("AddElementToElement:  index out of range");

///	ElemType us = beta ? beta * c(ci, cj) : (ElemType)0; // do not multiply if beta is 0, could be a NaN
///	us += a(ai, aj);
///	c(ci, cj) = us;
///}

////c[ci,cj] += a[ai,aj]
////template<class ElemType>
////void CPUMatrix<ElemType>::AddLogElementToElement(const CPUMatrix<ElemType>& a, size_t ai, size_t aj, CPUMatrix<ElemType>& c, size_t ci, size_t cj)
////{
////    if (ai >= a.GetNumRows() || aj >=a.GetNumCols() ||
////        ci >= c.GetNumRows() || cj >=c.GetNumCols())
////        InvalidArgument("AddElementToElement:  index out of range");
////
////    ElemType v = a(ai,aj);
////    c(ci, cj) += ((v < EPS_IN_LOG) ? LOG_EPS_IN_LOG : log(v));
////}

///#if 0 // now done as AddElementToElement (beta=0)
//// c[ci,cj] = a[ai,aj]
///template <class ElemType>
///void CPUMatrix<ElemType>::AssignElementToElement(const CPUMatrix<ElemType>& a, size_t ai, size_t aj, CPUMatrix<ElemType>& c, size_t ci, size_t cj)
///{
///	if (ai >= a.GetNumRows() || aj >= a.GetNumCols() ||
///		ci >= c.GetNumRows() || cj >= c.GetNumCols())
///		InvalidArgument("AssignElementToElement:  index out of range");

///	c(ci, cj) = a(ai, aj);
///}
///#endif

//	c += alpha * (a-b)		(a, b, c  must have same dim)
//		alpha	coefficient
//		a		Input matrix
//		b		Input matrix
//		c		Resulting matrix, user is responsible for allocating this
///template <class ElemType>
///void CPUMatrix<ElemType>::AddScaledDifference(const CPUMatrix<ElemType>& alpha, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c)
///{
///	if (alpha.GetNumElements() != 1)
///		InvalidArgument("AddScaledDifference:  alpha must be a 1x1 matrix");

///	AddScaledDifference(alpha(0, 0), a, b, c);
///}

//	 c = alpha * (a-b)		(a, b, c  must have same dim)
//		alpha	coefficient
//		a		Input matrix
//		b		Input matrix
//		c		Resulting matrix, user is responsible for allocating this
///template <class ElemType>
///void CPUMatrix<ElemType>::AssignScaledDifference(const CPUMatrix<ElemType>& alpha, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c)
///{
///	if (alpha.GetNumElements() != 1)
///		InvalidArgument("AddScaledDifference:  alpha must be a 1x1 matrix");

///	AssignScaledDifference(alpha(0, 0), a, b, c);
///}

// treat matrices as vectors. do vec(a)^T vec(b)
template <class ElemType>
ElemType CPUMatrix<ElemType>::InnerProductOfMatrices(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b)
{
	if (a.IsEmpty() || b.IsEmpty())
		LogicError("InnerProductOfMatrices:  one of the input matrices is empty");

	const int m = (int)a.GetNumRows();
	const int n = (int)a.GetNumCols();
	const int k = (int)b.GetNumRows();
	const int l = (int)b.GetNumCols();

	assert(m > 0 && n > 0 && k > 0 && l > 0); // converting from size_t to int may cause overflow
	if (m != k || n != l)
		InvalidArgument("InnerProductOfMatrices: Matrices a and b should have same dimension");

	if (std::is_same<ElemType, double>::value)
	{
		return (ElemType) cblas_ddot((int)a.GetNumElements(), reinterpret_cast<double*>(a.GetData()), 1, reinterpret_cast<double*>(b.GetData()), 1);
	}
	else if (std::is_same<ElemType, float>::value)
	{
#pragma warning(suppress : 4244)
		return (ElemType) cblas_sdot((int)a.GetNumElements(), reinterpret_cast<float*>(a.GetData()), 1, reinterpret_cast<float*>(b.GetData()), 1);
	}
	else
	{
		RuntimeError("InnerProductOfMatrices; Unsupported data type");
	}
}

// c = .power(a alpha)
template <class ElemType>
void CPUMatrix<ElemType>::ElementWisePower(ElemType alpha, const CPUMatrix<ElemType>& a, CPUMatrix<ElemType>& c)
{
	if (a.IsEmpty()) LogicError("ElementWisePower; A[%lu,%lu] is empty", a.GetNumRows(), a.GetNumCols());

	if (c.GetNumRows()!=a.GetNumRows() || c.GetNumCols()!=a.GetNumCols()) c.Resize(a.GetNumRows(), a.GetNumCols());
	if (alpha == 2)
	{
#pragma omp parallel for
		foreach_coord (i, j, c)
		{
			c(i, j) = a(i, j) * a(i, j);
		}
	}
	else if (alpha == 3)
	{
#pragma omp parallel for
		foreach_coord (i, j, c)
		{
			c(i, j) = a(i, j) * a(i, j) * a(i, j);
		}
	}
	else
	{
//#pragma omp parallel for
		foreach_coord (i, j, c)
		{
			c(i, j) = pow(a(i, j), alpha);
		}
	}
}

template <class ElemType>
void CPUMatrix<ElemType>::BatchMatMul(ElemType beta, const CPUMatrix<ElemType>& a, bool transposeA, int m, const CPUMatrix<ElemType>& b, bool transposeB, int n, CPUMatrix<ElemType>& c, bool isColWise)
{
	if (a.IsEmpty() || b.IsEmpty())
		LogicError("BatchMatMul; A[%lu,%lu] or B[%lu,%lu] is empty", a.GetNumRows(), a.GetNumCols(), b.GetNumRows(), b.GetNumCols());

	if (!isColWise) LogicError("BatchMatMul; Only column wise is supported");

	const int aSampleElemNum = (int)a.GetNumRows();
	const int aBatchSize = (int)a.GetNumCols();
	const int bSampleElemNum = (int)b.GetNumRows();
	const int bBatchSize = (int)b.GetNumCols();

	if (aBatchSize != bBatchSize)
		InvalidArgument("BatchMatMul; Matrices A[,%lu] and B[,%lu] have different size", a.GetNumCols(), b.GetNumCols());

	int k = aSampleElemNum / m;
	int kb = bSampleElemNum / n;

	if (k != kb)
		InvalidArgument("BatchMatMul; Matrices A's cols number should match matrices B's rows number");

	size_t cSampleElemNum = m * n;

	if (beta == 0) c.Resize(cSampleElemNum, aBatchSize);
	else c.VerifySize(cSampleElemNum, aBatchSize, "BatchMatMul");

#ifdef USE_OPENBLAS

	CBLAS_TRANSPOSE blasTransA = transposeA ? CblasTrans : CblasNoTrans;
	CBLAS_TRANSPOSE blasTransB = transposeB ? CblasTrans : CblasNoTrans;
	int lda = transposeA ? k : m;
	int ldb = transposeB ? n : k;
	int ldc = m;

	std::vector<const ElemType *> a_array; a_array.reserve(aBatchSize);
	std::vector<const ElemType *> b_array; b_array.reserve(aBatchSize);
	std::vector<ElemType *> c_array; c_array.reserve(aBatchSize);

	ElemType* aBufPtr = a.GetData();
	ElemType* bBufPtr = b.GetData();
	ElemType* cBufPtr = c.GetData();
	for (size_t i = 0; i < aBatchSize; i++)
	{
		a_array.push_back(aBufPtr + a.ColumnPos(i));
		b_array.push_back(bBufPtr + b.ColumnPos(i));
		c_array.push_back(cBufPtr + c.ColumnPos(i));
	}
	for (size_t i = 0; i < aBatchSize; i++)
	{
		if (sizeof(ElemType) == sizeof(double))
		{
			double alpha = 1.0;
			cblas_dgemm((CBLAS_ORDER)(int)MatrixOrder::ColMajor, blasTransA, blasTransB, m, n, k, alpha, reinterpret_cast<const double*>(a_array[i]), lda, reinterpret_cast<const double*>(b_array[i]), ldb, double(beta), reinterpret_cast<double*>(c_array[i]), ldc);
		}
		else
		{
			float alpha = 1.0f;
			cblas_sgemm((CBLAS_ORDER)(int)MatrixOrder::ColMajor, blasTransA, blasTransB, m, n, k, alpha, reinterpret_cast<const float*>(a_array[i]), lda, reinterpret_cast<const float*>(b_array[i]), ldb, float(beta), reinterpret_cast<float*>(c_array[i]), ldc);
		}
	}

#else

	std::vector<int> m_array(aBatchSize, m);
	std::vector<int> n_array(aBatchSize, n);
	std::vector<int> k_array(aBatchSize, k);
	std::vector<int> lda_array(aBatchSize, transposeA ? k : m);
	std::vector<int> ldb_array(aBatchSize, transposeB ? n : k);
	std::vector<int> ldc_array(aBatchSize, m);
	std::vector<int> group_size(1, aBatchSize);
	std::vector<CBLAS_TRANSPOSE> transa_array(aBatchSize, transposeA ? CblasTrans : CblasNoTrans);
	std::vector<CBLAS_TRANSPOSE> transb_array(aBatchSize, transposeB ? CblasTrans : CblasNoTrans);
	std::vector<const ElemType *> a_array; a_array.reserve(aBatchSize);
	std::vector<const ElemType *> b_array; b_array.reserve(aBatchSize);
	std::vector<ElemType *> c_array; c_array.reserve(aBatchSize);
	
	ElemType* aBufPtr = a.GetData();
	ElemType* bBufPtr = b.GetData();
	ElemType* cBufPtr = c.GetData();
	for (size_t i = 0; i < aBatchSize; i++)
	{
		a_array.push_back(aBufPtr + a.ColumnPos(i));
		b_array.push_back(bBufPtr + b.ColumnPos(i));
		c_array.push_back(cBufPtr + c.ColumnPos(i));
	}
	if (sizeof(ElemType) == sizeof(double))
	{
		std::vector<double> alpha_array(group_size[0], 1.0);
		std::vector<double> beta_array(group_size[0], double(beta));
		cblas_dgemm_batch(CblasColMajor, &transa_array[0], &transb_array[0], &m_array[0], &n_array[0], &k_array[0], &alpha_array[0],
						reinterpret_cast<const double**>(&a_array[0]), &lda_array[0], reinterpret_cast<const double**>(&b_array[0]), &ldb_array[0], &beta_array[0],
						reinterpret_cast<double**>(&c_array[0]), &ldc_array[0], 1, &group_size[0]);
	}
	else
	{
		std::vector<float> alpha_array(group_size[0], 1.0f);
		std::vector<float> beta_array(group_size[0], float(beta));
		cblas_sgemm_batch(CblasColMajor, &transa_array[0], &transb_array[0], &m_array[0], &n_array[0], &k_array[0], &alpha_array[0],
						reinterpret_cast<const float**>(&a_array[0]), &lda_array[0], reinterpret_cast<const float**>(&b_array[0]), &ldb_array[0], &beta_array[0],
						reinterpret_cast<float**>(&c_array[0]), &ldc_array[0], 1, &group_size[0]);
	}

#endif
}

///template <class ElemType>
///bool CPUMatrix<ElemType>::AreEqual(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, ElemType threshold /*= 1e-8*/)
///{
///	if (a.GetNumRows() != b.GetNumRows() || a.GetNumCols() != b.GetNumCols()) return false;
///#pragma omp parallel for
///	foreach_coord (i, j, a)
///		if (abs(a(i, j) - b(i, j)) > threshold) return false;
///	return true;
///}

//// see Matrix<ElemType>::TensorShuffleScaleAndAdd() for comments
///template <class ElemType>
///void CPUMatrix<ElemType>::TensorShuffleScaleAndAdd(ElemType keepWeight, const CPUMatrix<ElemType>& a, size_t D, size_t S, size_t M, size_t K, size_t T, ElemType scaleFactor, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c)
///{
///	size_t N = D * S * M * K * T;
///	const auto pa = a.GetData();
///	const auto pb = b.GetData();
///	auto pc = c.GetData();
///	// Note: This code is written to match a GPU implementation. It is not super-efficient on the CPU.
///	for (size_t na = 0; na < N; na++) // loop over all elements
///	{
///		// recover the 5 indices from the loop counter
///		size_t d = na % D;
///		size_t s = (na / D) % S;
///		size_t m = (na / D / S) % M;
///		size_t k = (na / D / S / M) % K;
///		size_t t = (na / D / S / M / K) % T;
///		// compute index for the a and b/c tensors
///		assert(na == (((t * K + k) * M + m) * S + s) * D + d); // input tensor of dimension (D x S x M x K x T)
///		size_t nb = (((t * S + s) * M + m) * K + k) * D + d;   // output tensor of dimension (D x K x M x S x T): k/K and s/S swapped
///		assert(nb < N);
///		// perform the computation
///		ElemType cval = keepWeight ? keepWeight * pb[nb] : (ElemType)0; // if weight is 0 then don't bother to read memory (efficiency) or to multiply (NaN-safe)
///		cval += scaleFactor * pa[na];
///		pc[nb] = cval;
///	}
///}

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::Ones(size_t rows, size_t cols)
{
	CPUMatrix<ElemType> c(rows, cols); // will initialize to 0
	c.SetValue((ElemType)1);
	return c;
}

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::Zeros(size_t rows, size_t cols)
{
	CPUMatrix<ElemType> c(rows, cols); // will initialize to 0
	c.SetValue((ElemType)0);
	return c;
}

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::Eye(size_t rows)
{
	CPUMatrix<ElemType> c(rows, rows); // will initialize to 0
	c.SetDiagonalValue((ElemType)1);
	return c;
}

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::RandomUniform(size_t rows, size_t cols, ElemType low, ElemType high, unsigned long seed)
{
	CPUMatrix<ElemType> c(rows, cols); // will initialize to 0
	c.SetUniformRandomValue(low, high, seed);
	return c;
}

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::RandomGaussian(size_t rows, size_t cols, ElemType mean, ElemType sigma, unsigned long seed)
{
	CPUMatrix<ElemType> c(rows, cols); // will initialize to 0
	c.SetGaussianRandomValue(mean, sigma, seed);
	return c;
}

///template <class ElemType>
///bool CPUMatrix<ElemType>::HasElement(const CPUMatrix<ElemType>& mat, ElemType v)
///{
///	bool bHas = false;

///	bool isvFinite = std::isfinite(v);
///#pragma omp parallel for
///	for (long j = 0; j < mat.GetNumElements(); j++)
///	{
///#pragma omp flush(bHas)
///		if (!bHas)
///		{
///			ElemType cur = mat.GetData()[j];
///			if (isvFinite && std::isfinite(cur))
///			{
///				if (cur == v)
///					bHas = true;
///			}
///			else if (std::isnan(v) && std::isnan(cur))
///				bHas = true;
///			else if (std::isinf(v) && std::isinf(cur) && std::signbit(v) == std::signbit(cur))
///				bHas = true;
///		}
///	}

///	return bHas;
///}

////        CPUMatrix<ElemType>& AssignElementProductOfWithShiftNeg(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, size_t shift, size_t negnumber);
// [this] = a .* b
//// here, a and b must be two row vectors of the same size, i.e. [1,m]
//// the inputs are two rwo vectors
//// the output is a matrix of size(neg+1, col)
///template <class ElemType>
///CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignElementProductOfWithShiftNeg(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, size_t shift, size_t negnumber)
///{
///	if (a.IsEmpty() || b.IsEmpty())
///		LogicError("AssignElementProductOfWithShiftNeg; Matrix is empty");

///	if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
///		InvalidArgument("AssignElementProductOfWithShiftNeg; The input matrix dimensions do not match");

///	if (a.GetNumRows() != 1)
///		InvalidArgument("AssignElementProductOfWithShiftNeg; The input matrix must be a row vector");

///	auto& us = *this;
///	if (this != &a)
///	{
///		Resize(negnumber+1, a.GetNumCols());
///		//            Resize(a.GetNumRows(), a.GetNumCols());
///	}

///	long m = (long) GetNumRows(), n = (long) GetNumCols(); // a and b are of size (1,n)
///	// #pragma omp parallel for

///	for (long j = 0; j < n; j++)
///	{
///		us(0, j) = a(0, j) * b(0, j);
///	}
///	for (long j = 0; j < n; j++)
///	{
///		for (long i = 1; i < m; i++)
///		{
///			us(i, j) = a(0, j) * b(0, (j + shift + i - 1) % n);
///		}
///	}

///	return *this;
///}

///template <class ElemType>
///void CPUMatrix<ElemType>::InnerProductWithShiftNeg(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c, bool isColWise, size_t shift, size_t negnumber)
///{
///	if (a.IsEmpty() || b.IsEmpty())
///		LogicError("InnerProduct:  one of the input matrices is empty");

///	const int m = (int)a.GetNumRows();
///	const int n = (int)a.GetNumCols();
///	const int k = (int)b.GetNumRows();
///	const int l = (int)b.GetNumCols();

///	assert(m > 0 && n > 0 && k > 0 && l > 0); // converting from size_t to int may cause overflow
///	if (m != k || n != l)
///		InvalidArgument("InnerProduct: Matrices a and b should have same dimension");

///	if ((isColWise && m == 1) || !isColWise && n == 1) // in this case it's equivalent to element-wise product
///	{
///		InvalidArgument("InnerProduct: Both matrices should be normal ones, not vectors");
///		//            c.AssignElementProductOf(a, b);
///	}
///	else if (isColWise) // col-wise
///	{
///		c.Resize(negnumber+1, n); // this line ischanged

///		ElemType* aBufPtr = a.GetData();
///		ElemType* bBufPtr = b.GetData();
///		if (std::is_same<ElemType, double>::value)
///		{
///			for (long j = 0; j < n; j++)
///			{
///				c(0, j) = (ElemType) cblas_ddot(m, reinterpret_cast<double*>(aBufPtr + a.ColumnPos(j)), 1, reinterpret_cast<double*>(bBufPtr + b.ColumnPos(j)), 1);
///			}
///			for (long j = 0; j < n; j++)
///			{
///				for (long i = 1; i < negnumber + 1; i++)
///				{
///					c(i, j) = (ElemType) cblas_ddot(m, reinterpret_cast<double*>(aBufPtr + a.ColumnPos(j)), 1, reinterpret_cast<double*>(bBufPtr + b.ColumnPos((j + shift + i - 1) % n)), 1);
///				}
///			}
///		}
///		else if (std::is_same<ElemType, float>::value)
///		{
///			for (long j = 0; j < n; j++)
///			{
///				c(0, j) = (ElemType) cblas_sdot(m, reinterpret_cast<float*>(aBufPtr + a.ColumnPos(j)), 1, reinterpret_cast<float*>(bBufPtr + b.ColumnPos(j)), 1);
///			}
///			for (long j = 0; j < n; j++)
///			{
///				for (long i = 1; i < negnumber + 1; i++)
///				{
///					c(i, j) = (ElemType) cblas_sdot(m, reinterpret_cast<float*>(aBufPtr + a.ColumnPos(j)), 1, reinterpret_cast<float*>(bBufPtr + b.ColumnPos((j + shift + i - 1) % n)), 1);
///				}
///			}
///		}
///		else
///		{
///			RuntimeError("InnerProductWithShiftNeg; Unsupported data type");
///		}
///	}
///	else
///	{
///		InvalidArgument("InnerProductWithShiftNeg; Rowwise is not supported yet");

///		c.Resize(m, 1);

///		ElemType* aBufPtr = a.GetData();
///		ElemType* bBufPtr = b.GetData();
///		if (std::is_same<ElemType, double>::value)
///		{
///#pragma omp parallel for
///			foreach_row (i, c)
///			{
///				c(i, 0) = (ElemType) cblas_ddot(n, reinterpret_cast<double*>(aBufPtr + i), m, reinterpret_cast<double*>(bBufPtr + i), m);
///			}
///		}
///		else if (std::is_same<ElemType, float>::value)
///		{
///#pragma omp parallel for
///			foreach_row (i, c)
///			{
///#pragma warning(suppress : 4244)
///				c(i, 0) = cblas_sdot(n, reinterpret_cast<float*>(aBufPtr + i), m, reinterpret_cast<float*>(bBufPtr + i), m);
///			}
///		}
///		else
///		{
///			RuntimeError("InnerProductWithShiftNeg; Unsupported data type");
///		}
///	}
///}

///template <class ElemType>
///CPUMatrix<ElemType>& CPUMatrix<ElemType>::GetARowByIndex(const CPUMatrix<ElemType>& a, size_t index)
///{
///	if (a.IsEmpty())
///		LogicError("GetARowByIndex:  the input matrices is empty");

///	const int m = (int)a.GetNumRows();
///	const int n = (int)a.GetNumCols();

///	if (index < 0 || index >= m)
///		LogicError("GetARowByIndex:  the row index is out of range");

///	assert(m > 0 && n > 0); // converting from size_t to int may cause overflow

///	auto& us = *this;
///	Resize(1, n);
///	for (long j = 0; j < n; j++)
///	{
///		us(0, j) = a(index, j);
///	}

///	return *this;
///}

//// input: a, a row vector
//// input: b, a matrix. b.col == a.col
//// input firstmatrixfixed: If true, keep a's order. Otherwise, keep b's order
//// output: c, a matrix. c.size == b.size
////*
///	Example, a = [a1 a2 a3]
///	b = [b11 b12 b13;
///	b21 b22 b23 ]

///	if true:
///	shift = 1

///	then c = [a1*b12 a2*b13 a3*b11
///	a1*b22 a2*b23 a3*b21]

///	if shift = 2
///	then c = [  a1*b13 a2*b11 a3*b12
///	a1*b23 a2*b21 a3*b22]
///	i.e. we do column-wise shift

///	if false:
///	shift = 1

///	then c = [a2*b11 a3*b12 a1*b13
///	a2*b21 a3*b22 a1*b23]

///	shift = 2

///	then c = [  a3*b11 a1*b12 a2*b13
///	a3*b21 a1*b22 a2*b23]


///	*/
///template <class ElemType>
///void CPUMatrix<ElemType>::ConductRowElementMultiplyWithShift(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c, size_t shift, bool bFirstmatrixfixed)
///{
///	if (a.IsEmpty() || b.IsEmpty())
///		LogicError("InnerProduct:  one of the input matrices is empty");

///	const int m = (int)a.GetNumRows();
///	const int n = (int)a.GetNumCols();
///	const int k = (int)b.GetNumRows();
///	const int l = (int)b.GetNumCols();

///	assert(m > 0 && n > 0 && k > 0 && l > 0); // converting from size_t to int may cause overflow
///	if (m != 1 || n != l)
///		InvalidArgument("InnerProduct: Matrices a and b should have same dimension");

///	c.Resize(k, l); // c must the same size of b

///	if (bFirstmatrixfixed)
///	{
///		for (long j = 0; j < l; j++)
///		{
///			for (long i = 0; i < k; i++)
///			{
///				c(i, j) = a(0, j) * b(i, (j + shift) % l);
///			}
///		}
///	}
///	else
///	{
///		for (long j = 0; j < l; j++)
///		{
///			for (long i = 0; i < k; i++)
///			{
///				c(i, j) = a(0, (j + shift) % l) * b(i, j);
///			}
///		}
///	}
///}

////        CPUMatrix<ElemType>& AssignElementProductOfWithShift(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, size_t shift);
// [this] = a .* b
//// here, a and b must be two row vectors of the same size, i.e. [1,m]. We will do element product with shift.
//// inputs are 2 row vectors
//// output is a row vector
///template <class ElemType>
///CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignElementProductOfWithShift(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, size_t shift)
///{
///	if (a.IsEmpty() || b.IsEmpty())
///		LogicError("AssignElementProductOfWithShiftNeg; Matrix is empty");

///	if (a.GetNumRows() != b.GetNumRows() || a.GetNumCols() != b.GetNumCols())
///		InvalidArgument("AssignElementProductOfWithShiftNeg; The input matrix dimensions do not match");

///	if (a.GetNumRows() != 1)
///		InvalidArgument("AssignElementProductOfWithShiftNeg; The input matrix must be a row vector");

///	auto& us = *this;
///	if (this != &a)
///	{
///		Resize(1, a.GetNumCols());
///		//            Resize(a.GetNumRows(), a.GetNumCols());
///	}

///	// long m = (long) GetNumRows(), n = (long) GetNumCols();  // a and b are of size (1,n)
///	long n = (long) GetNumCols(); // a and b are of size (1,n)
///#pragma omp parallel for
///	for (long j = 0; j < n; j++)
///	{
///		us(0, j) = a(0, j) * b(0, (j + shift) % n);
///	}
///	return *this;
///}

//// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

//// 'double' version of LogAdd
///inline double LogAddD(double x, double y)
///{
///	return LogAdd(x, y);
///}

///template <class ElemType>
///ElemType CPUMatrix<ElemType>::LogSumOfElements() const
///{
///	ElemType fAlpha = (ElemType) LZERO;
///	ElemType* bufPtr = GetData();
///	for (int k = 0; k < GetNumElements(); k++)
///		fAlpha = (ElemType) LogAddD(fAlpha, bufPtr[k]);
///	return fAlpha;
///}

///template <class ElemType>
///void CPUMatrix<ElemType>::RCRFBackwardCompute(const CPUMatrix<ElemType>& alpha, CPUMatrix<ElemType>& beta,
///												const CPUMatrix<ElemType>& lbls,
///												const CPUMatrix<ElemType>& pair_scores)
///{
///	int iNumPos = (int)lbls.GetNumCols();
///	int iNumLab = (int)lbls.GetNumRows();

///	int lastLbl = -1;
///	for (int ik = 0; ik < lbls.GetNumRows(); ik++)
///		if (lbls(ik, iNumPos - 1) != 0)
///		{
///			lastLbl = ik;
///			break;
///		}

///	beta.Resize(iNumLab, iNumPos);

///	for (int t = iNumPos - 1; t >= 0; t--)
///	{
///#pragma omp parallel for
///		for (int k = 0; k < iNumLab; k++)
///		{
///			_rcrfBackwardCompute(t, k, alpha, beta, pair_scores);
///		}
///	}
///};

//// Calculate alpha in forward-backward calculation. equation (6), (7) in ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf
//// GPU x dimension corresponds to utterances, y dimension corresponds to phone sequence in each utterance
//// prob (input): the posterior output from the network
//// alpha (output): alpha for forward-backward calculation.
//// phoneSeq (input): phone ID sequence for each utterance in this minibatch, each col is one utterance
//// phoneBound (input): phone boundary (frame index) of each phone for each utterance in this minibatch, each col is one utterance
//// uttToChanInd (input):  map from utterance ID to minibatch channel ID. We need this because each channel may contain more than one utterance.
//// uttFrameNum (input): the frame number of each utterance. The size of this vector =  the number of all utterances in this minibatch
//// uttBeginFrame(input): the position of the first frame of each utterance in the minibatch channel. We need this because each channel may contain more than one utterance.
//// uttPhoneNum (input): the phone number of each utterance. The size of this vector =  the number of all utterances in this minibatch
//// numChannels (input): channel number in this minibatch
//// uttNum (input): number of utterances
//// t (input): time stamp to process
//// maxPhoneNum (input): the max number of phones between utterances
//// totalPhoneNum (input): the total number of phones of all utterances
//// blankTokenId (input): id of the CTC blank token
//// delayConstraint -- label output delay constraint introduced during training that allows to have shorter delay during inference.
////      Alpha and Beta scores outside of the delay boundary are set to zero.
////      Setting this parameter smaller will result in shorted delay between label output during decoding.
////      delayConstraint=-1 means no constraint
///template<class ElemType>
///void _assignAlphaScore(
///	const ElemType *prob,
///	ElemType *alphaScore,
///	ElemType *phoneSeq,
///	ElemType *phoneBound,
///	const std::vector<size_t>& uttToChanInd,
///	const std::vector<size_t>& uttFrameNum,
///	const std::vector<size_t>& uttBeginFrame,
///	const std::vector<size_t>& uttPhoneNum,
///	size_t numChannels,
///	const size_t uttNum,
///	const size_t  t,
///	const size_t maxPhoneNum, // Maximum length of utterance in this MB
///	const size_t totalPhoneNum, // Total number of phones
///	const size_t blankTokenId,
///	const int delayConstraint)
///{
///	for (size_t uttId = 0;uttId < uttNum;uttId++) {

///		// Number of phones and frames in this utterance
///		size_t frameNum = uttFrameNum[uttId];
///		if (t >= frameNum) continue;

///		size_t phoneNum = uttPhoneNum[uttId];

///#pragma omp parallel for
///		for (int phoneSeqId = 1;phoneSeqId < phoneNum - 1;phoneSeqId++) {
///			// Index of the label in the sequence

///			// Current and previous phone indices in phoneSeq matrix
///			size_t labelid = uttId*maxPhoneNum + phoneSeqId;

///			// Actual current phone label
///			size_t phoneId = (size_t)(phoneSeq[labelid]);

///			// Index of the current frame in minibatch
///			size_t timeId = (t + uttBeginFrame[uttId])*numChannels + uttToChanInd[uttId];

///			// Index of probability of observing phoneId at frame timeId
///			size_t probId = timeId*totalPhoneNum + phoneId;

///			size_t alphaId = maxPhoneNum* timeId + phoneSeqId; // alpha_t(s)

///			if (t == 0)
///			{
///				// Initialize recursion
///				if (phoneSeqId == 1 || phoneSeqId == 2)
///				{
///					alphaScore[alphaId] = prob[probId];
///				}
///			}
///			else
///			{
///				if (phoneSeqId >= 1)
///				{
///					size_t timeId_1 = timeId - numChannels; // Index corresponding to (t-1)
///					size_t alphaId_0 = maxPhoneNum* timeId_1 + phoneSeqId; // alpha_{t-1}(s)
///					size_t alphaId_1 = alphaId_0 - 1; // alpha_{t-1}(s-1)
///					size_t alphaId_2 = alphaId_0 - 2; // alpha_{t-1}(s-2)
///					ElemType x = (ElemType)LZERO;

///					ElemType ascore;
///					if (phoneSeqId > 2)
///					{
///						size_t labelid_2 = labelid - 2;
///						// if current label is not blank and not equal prev non-blank label
///						if ((size_t)(phoneSeq[labelid]) != blankTokenId && phoneId != (size_t)(phoneSeq[labelid_2]))
///						{
///							x = LogAdd(x, alphaScore[alphaId_2]);
///						}
///					}

///					if (phoneSeqId > 1)
///					{
///						x = LogAdd(x, alphaScore[alphaId_1]);
///					}

///					x = LogAdd(x, alphaScore[alphaId_0]);

///					if (phoneId != SIZE_MAX)
///						ascore = prob[probId]; // Probability of observing given label at given time
///					else
///						ascore = 0;
///					alphaScore[alphaId] = (ElemType)x + ascore;
///					if (delayConstraint != -1)
///					{
///						size_t labelid_r = labelid + 2;
///						size_t phoneBoundId_r = (size_t)(phoneBound[labelid_r]);
///						if (phoneId == blankTokenId)
///						{
///							// only constraint right side
///							if (t > phoneBoundId_r + delayConstraint - 1)
///								alphaScore[alphaId] = (ElemType)LZERO;
///						}
///						else if (phoneId != blankTokenId)
///						{
///							if (t > phoneBoundId_r + delayConstraint)
///								alphaScore[alphaId] = (ElemType)LZERO;
///						}
///					}
///				}

///			}
///		}
///	}
///}

//// Calculate beta in forward-backward calculation, equation (10), (11) in ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf
//// See _assignAlphaScore for the explanation of parameters
///template<class ElemType>
///void _assignBetaScore(
///	const ElemType *prob,
///	ElemType *betaScore,
///	ElemType *phoneSeq,
///	ElemType *phoneBound,
///	const std::vector<size_t>& uttToChanInd,
///	const std::vector<size_t>& uttFrameNum,
///	const std::vector<size_t>& uttBeginFrame,
///	const std::vector<size_t>& uttPhoneNum,
///	const size_t numChannels,
///	const size_t uttNum,
///	const long  t,
///	const size_t maxPhoneNum,
///	const size_t totalPhoneNum,
///	const size_t blankTokenId,
///	const int delayConstraint)
///{
///	for (size_t uttId = 0;uttId < uttNum;uttId++) {

///		// Number of phones and frames in this utterance
///		size_t frameNum = uttFrameNum[uttId];
///		if (t >= frameNum) continue;

///		size_t phoneNum = uttPhoneNum[uttId];

///#pragma omp parallel for
///		for (int phoneSeqId = 1;phoneSeqId < phoneNum - 1;phoneSeqId++) {

///			size_t labelid = uttId*maxPhoneNum + phoneSeqId;
///			size_t labelid_2 = labelid + 2;
///			size_t phoneId = (LONG64)(phoneSeq[labelid]);
///			size_t timeId = (t + uttBeginFrame[uttId])*numChannels + uttToChanInd[uttId];
///			size_t probId = timeId*totalPhoneNum + phoneId;
///			size_t betaid = maxPhoneNum* timeId + phoneSeqId;
///			size_t timeId_1 = timeId + numChannels;
///			size_t betaid_0 = maxPhoneNum* timeId_1 + phoneSeqId;
///			size_t betaid_1 = betaid_0 + 1;
///			size_t betaid_2 = betaid_0 + 2;

///			if (t == frameNum - 1)
///			{
///				if (phoneSeqId == phoneNum - 3 || phoneSeqId == phoneNum - 2)
///				{
///					betaScore[betaid] = prob[probId];
///				}
///			}
///			else
///			{
///				if (phoneSeqId >= 1)
///				{
///					ElemType x = (ElemType)LZERO;
///					ElemType ascore;
///					if (phoneSeqId < phoneNum - 3)
///					{
///						if (phoneSeq[labelid] != blankTokenId && phoneId != phoneSeq[labelid_2])
///						{
///							x = LogAdd(x, betaScore[betaid_2]);
///						}
///					}

///					if (phoneSeqId < phoneNum - 2)
///					{
///						x = LogAdd(x, betaScore[betaid_1]);
///					}

///					x = LogAdd(x, betaScore[betaid_0]);

///					if (phoneId != SIZE_MAX)
///						ascore = prob[probId];
///					else
///						ascore = 0;
///					betaScore[betaid] = (ElemType)x + ascore;
///					if (delayConstraint != -1)
///					{
///						size_t phoneBoundId_r = (size_t)(phoneBound[labelid_2]);
///						if (phoneId == blankTokenId)
///						{
///							if (t > phoneBoundId_r + delayConstraint - 1)
///								betaScore[betaid] = (ElemType)LZERO;
///						}
///						else if (phoneId != blankTokenId)
///						{
///							if (t > phoneBoundId_r + delayConstraint)
///								betaScore[betaid] = (ElemType)LZERO;
///						}
///					}
///				}
///			}
///		}
///	}
///}

//// Calculate CTC score. equation (8) in ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf
///template<class ElemType>
///void _assignTotalScore(ElemType *betaScore,
///	std::vector<ElemType>& totalScore,
///	const size_t uttNum,
///	const std::vector<size_t>& uttToChanInd,
///	const std::vector<size_t>& uttBeginFrame,
///	const size_t numChannels,
///	const size_t maxPhoneNum)
///{
///#pragma omp parallel for
///	for (int uttId = 0; uttId < uttNum; uttId++) {
///		if (uttId < uttNum)
///		{
///			LONG64 alphaId_0 = (uttBeginFrame[uttId] * numChannels + uttToChanInd[uttId]) * maxPhoneNum;

///			betaScore[alphaId_0] = LogAdd(betaScore[alphaId_0 + 1], betaScore[alphaId_0 + 2]);
///			totalScore[uttId] = betaScore[alphaId_0];
///		}
///	}
///}

//// Calculate derivative, equation (15) in ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf
//// See _assignAlphaScore for the explanation of parameters
///template<class ElemType>
///void _assignCTCScore(
///	ElemType *CTCscore,
///	ElemType *prob,
///	ElemType *alphaScore,
///	ElemType *betaScore,
///	ElemType *phoneSeq,
///	const size_t uttNum,
///	const std::vector<size_t>& uttToChanInd,
///	const std::vector<size_t>& uttBeginFrame,
///	const std::vector<size_t>& uttPhoneNum,
///	const std::vector<size_t>& uttFrameNum,
///	const size_t numChannels,
///	const size_t maxPhoneNum,
///	const size_t totalPhoneNum)
///{
///	for (size_t uttId = 0;uttId < uttNum;uttId++) {
///#pragma omp parallel for
///		for (int t = 0; t < uttFrameNum[uttId]; t++) {
///			size_t phoneNum = uttPhoneNum[uttId];
///			size_t alphaId_0 = (uttBeginFrame[uttId] * numChannels + uttToChanInd[uttId]) * maxPhoneNum;
///			size_t timeId = (t + uttBeginFrame[uttId])*numChannels + uttToChanInd[uttId];
///			ElemType P_lx = betaScore[alphaId_0];

///			for (int s = 1; s < phoneNum - 1; s++)
///			{
///				long phoneId = phoneSeq[uttId*maxPhoneNum + s];
///				size_t alphaId = maxPhoneNum* timeId + s;
///				size_t probId = timeId*totalPhoneNum + phoneId;

///				if (phoneId != SIZE_MAX)
///				{
///					ElemType logoccu = alphaScore[alphaId] + betaScore[alphaId] - prob[probId] - (ElemType)P_lx;
///					CTCscore[probId] = LogAdd(CTCscore[probId], logoccu);
///				}
///			}

///			for (int s = 0; s < totalPhoneNum; s++)
///			{
///				size_t probId = timeId*totalPhoneNum + s;
///				ElemType logoccu = CTCscore[probId];
///				if (logoccu < LZERO)
///					CTCscore[probId] = 0.0f;
///				else
///					CTCscore[probId] = exp(logoccu);
///			}
///		}
///	}
///}

///template<class ElemType>
///CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignCTCScore(
///	const CPUMatrix<ElemType>& prob, CPUMatrix<ElemType>& alpha, CPUMatrix<ElemType>& beta,
///	const CPUMatrix<ElemType>& phoneSeq, const CPUMatrix<ElemType>& phoneBoundary, CPUMatrix<ElemType> & totalScore, const std::vector<size_t>& uttToChanInd, const std::vector<size_t> & uttBeginFrame, const std::vector<size_t> & uttFrameNum,
///	const std::vector<size_t> & uttPhoneNum, size_t numParallelSequences, size_t maxFrameNum, size_t blankTokenId, const int delayConstraint, bool isColWise)
///{
///	// Column wise representation of sequences in input matrices (each column is one sequence/utterance)
///	if (isColWise)
///	{
///		// Total number of phones
///		size_t totalPhoneNum = prob.GetNumRows();
///		size_t uttNum = uttFrameNum.size();

///		// Max number of phones in utterances in this minibatch
///		size_t maxPhoneNum = phoneSeq.GetNumRows();

///		for (size_t t = 0; t < maxFrameNum; t++)
///		{
///			_assignAlphaScore(prob.GetData(), alpha.GetData(), phoneSeq.GetData(), phoneBoundary.GetData(), uttToChanInd,
///				uttFrameNum, uttBeginFrame, uttPhoneNum, numParallelSequences, uttNum, t, maxPhoneNum, totalPhoneNum, blankTokenId, delayConstraint);
///		}

///		for (LONG64 t = maxFrameNum - 1; t >= 0; t--)
///		{
///			_assignBetaScore(prob.GetData(), beta.GetData(), phoneSeq.GetData(), phoneBoundary.GetData(), uttToChanInd,
///				uttFrameNum, uttBeginFrame, uttPhoneNum, numParallelSequences, uttNum, t, maxPhoneNum, totalPhoneNum, blankTokenId, delayConstraint);
///		}

///		std::vector<ElemType> scores(uttNum);
///		_assignTotalScore(beta.GetData(), scores, uttNum, uttToChanInd, uttBeginFrame, numParallelSequences, maxPhoneNum);

///		_assignCTCScore(GetData(), prob.GetData(), alpha.GetData(), beta.GetData(), phoneSeq.GetData(), uttNum, uttToChanInd,
///			uttBeginFrame, uttPhoneNum, uttFrameNum, numParallelSequences, maxPhoneNum, totalPhoneNum);

///		totalScore(0, 0) = 0.0;
///		for (size_t utt = 0; utt < uttNum; utt++)
///		{
///			totalScore(0,0) -= scores[utt];
///		}

///		return *this;

///	}
///	else {
///		LogicError("Only ColWise minibatch layout is supported");
///	}

///	return *this;
///}

//// the kernel function for RCRF backward computation
///template <class ElemType>
///void CPUMatrix<ElemType>::_rcrfBackwardCompute(size_t t, size_t k, const CPUMatrix<ElemType>& alpha,
///												CPUMatrix<ElemType>& beta,
///												const CPUMatrix<ElemType>& pair_scores)
///{
///	size_t iNumLab = alpha.GetNumRows();
///	size_t iNumPos = alpha.GetNumCols();

///	ElemType fSum;
///	ElemType fTmp = (ElemType) LZERO;
///	if (t == iNumPos - 1)
///	{
///		fSum = (ElemType) LZERO;
///		for (int j = 0; j < iNumLab; j++)
///		{
///			fSum = (ElemType) LogAddD(fSum, alpha(j, t));
///		}

///		fTmp = alpha(k, t) - fSum;
///		beta(k, t) = fTmp;
///	}
///	else
///	{
///		for (int j = 0; j < iNumLab; j++)
///		{
///			fSum = (ElemType) LZERO;
///			for (int m = 0; m < iNumLab; m++)
///			{
///				fSum = (ElemType) LogAddD(fSum, alpha(m, t) + pair_scores(j, m));
///			}

///			fTmp = (ElemType) LogAddD(fTmp, beta(j, t + 1) + alpha(k, t) + pair_scores(j, k) - fSum);
///		}
///		beta(k, t) = fTmp;
///	}
///}

///template <class ElemType>
///void CPUMatrix<ElemType>::RCRFTransGrdCompute(const CPUMatrix<ElemType>& lbls,
///												const CPUMatrix<ElemType>& alpha,
///												const CPUMatrix<ElemType>& beta,
///												const CPUMatrix<ElemType>& pair_scores,
///												CPUMatrix<ElemType>& grd)
///{
///	int iNumPos = (int)alpha.GetNumCols();
///	int iNumLab = (int)alpha.GetNumRows();

///	int firstLbl = -1;
///	for (int ik = 0; ik < lbls.GetNumRows(); ik++)
///		if (lbls(ik, 0) != 0)
///		{
///			firstLbl = ik;
///			break;
///		}

///	for (size_t tPos = 0; tPos < iNumPos; tPos++)
///	{
///		CPUMatrix<ElemType> b = beta.GetColumnSlice(tPos, 1);
///		CPUMatrix<ElemType> a;
///		if (tPos > 0)
///			a = alpha.GetColumnSlice(tPos - 1, 1);

///#pragma omp parallel for
///		for (int i = 0; i < iNumLab; i++)
///		{
///			_rcrfTransGrdCompute(i, lbls, alpha, beta, pair_scores, grd, tPos);
///		}

///		// transition score
///		int i = -1;
///		if (tPos == 0)
///			i = firstLbl;
///		else
///		{
///			for (int ik = 0; ik < lbls.GetNumRows(); ik++)
///				if (lbls(ik, tPos - 1) != 0)
///				{
///					i = ik;
///					break;
///				}
///		}

///		int j = -1;
///		for (int ik = 0; ik < lbls.GetNumRows(); ik++)
///		{
///			if (lbls(ik, tPos) != 0)
///			{
///				j = ik;
///				break;
///			}
///		}

///		grd(j, i) -= 1.0;
///	}
///};

///template <class ElemType>
///void CPUMatrix<ElemType>::_rcrfTransGrdCompute(size_t i,
///												const CPUMatrix<ElemType>& lbls,
///												const CPUMatrix<ElemType>& alpha,
///												const CPUMatrix<ElemType>& beta,
///												const CPUMatrix<ElemType>& pair_scores,
///												CPUMatrix<ElemType>& grd,
///												const size_t tPos // position
///												)
///{
///	int iNumLab = (int)alpha.GetNumRows();

///	int firstLbl = -1;
///	for (int ik = 0; ik < lbls.GetNumRows(); ik++)
///		if (lbls(ik, 0) != 0) { firstLbl = ik; break; }

///	CPUMatrix<ElemType> b = beta.GetColumnSlice(tPos, 1);
///	CPUMatrix<ElemType> a;
///	if (tPos > 0) a = alpha.GetColumnSlice(tPos - 1, 1);

///	{
///		ElemType fTmp = (ElemType) LZERO;
///		for (int j = 0; j < iNumLab; j++)
///		{
///			if (tPos) fTmp = a(i, 0);
///			else if (i == firstLbl) fTmp = 0;
///			else fTmp = (ElemType) LZERO;
///			fTmp += pair_scores(j, i);

///			ElemType fSum = (ElemType) LZERO;
///			for (int k = 0; k < iNumLab; k++)
///			{
///				ElemType fTmp2;
///				if (tPos) fTmp2 = a(k, 0);
///				else if (k == firstLbl) fTmp2 = 0;
///				else fTmp2 = (ElemType) LZERO;
///				fSum = (ElemType) LogAddD(fSum, fTmp2 + pair_scores(j, k));
///			}
///			fTmp -= fSum;
///			fTmp += b(j, 0);
///			grd(j, i) += exp(fTmp);
///		}
///	}
///};
///template <class ElemType>
///CPUMatrix<ElemType>& CPUMatrix<ElemType>::DropFrame(const CPUMatrix<ElemType>& label, const CPUMatrix<ElemType>& gamma, const ElemType& threshhold)
///{
///	auto& us = *this;
///	if (us.GetNumCols() != gamma.GetNumCols() || us.GetNumRows() != gamma.GetNumRows())
///		LogicError("DropFrame: target matrix is not in the same size as gamm matrix");

///#pragma omp parallel for
///	foreach_column (j, label)
///	{

///		bool dropframe = false;
///		foreach_row (i, label)
///		{
///			if (fabs(label(i, j) - 1.0f) < 0.1)
///			{
///				if (gamma(i, j) < threshhold)
///					dropframe = true;
///				break;
///			}
///		}

///		foreach_row (i, label)
///		{
///			us(i, j) = 0.0f;
///		}
///	}

///	return *this;
///}

///template <class ElemType>
///CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignSequenceError(ElemType hsmoothingWeight, const CPUMatrix<ElemType>& label,
///																const CPUMatrix<ElemType>& dnnoutput, const CPUMatrix<ElemType>& gamma, ElemType alpha)
///{
///	auto& us = *this;
///	foreach_coord (i, j, us)
///		us(i, j) += alpha * (label(i, j) - (1 - hsmoothingWeight) * dnnoutput(i, j) - hsmoothingWeight * gamma(i, j));
///	return *this;
///}

//// note: this function does not depend on the <ElemType> parameter
///template <class ElemType>
///int CPUMatrix<ElemType>::SetNumThreads(int numThreads)
///{
///	if (numThreads == 0) return numThreads;	// use default

///	int mthreads = (int)std::thread::hardware_concurrency();
///	if (numThreads <= 0) numThreads = std::max(1, mthreads + numThreads);
///	if (numThreads > mthreads) numThreads = mthreads;

///#ifdef _OPENMP
///	omp_set_num_threads(numThreads);
///	numThreads = omp_get_max_threads();

///	#ifdef USE_MKL
///		mkl_set_num_threads(numThreads);
///	#elif defined(USE_OPENBLAS)
///		openblas_set_num_threads(numThreads);
///	#endif
///#endif
///	return numThreads;
///}

///template <class ElemType>
///int CPUMatrix<ElemType>::GetMaxNumThreads()
///{
///	int numThreads = (int)std::thread::hardware_concurrency();
///#ifdef _OPENMP
///	numThreads = omp_get_max_threads();
///#endif
///	return numThreads;
///}

// To ensure Intel MKL calls return the same results on all Intel or Intel compatible CPUs,
// the function set CBWR compatible mode.
template <class ElemType>
void CPUMatrix<ElemType>::SetCompatibleMode()
{
	// mkl_cbwr_set not supported in MKLML yet
	// Explanation on numeric diff: https://software.intel.com/en-us/articles/introduction-to-the-conditional-numerical-reproducibility-cnr
	// #ifdef USE_MKL
	//    if (mkl_cbwr_set(MKL_CBWR_COMPATIBLE) != MKL_CBWR_SUCCESS)
	//        RuntimeError("Could not set MKL compatible mode");
	// #endif
}

///template <class ElemType>
///void CPUMatrix<ElemType>::SetOptimizationFlags(int flags)
///{
///	m_optimizationFlags = flags;
///}

///template <class ElemType>
///int CPUMatrix<ElemType>::GetOptimizationFlags()
///{
///	return m_optimizationFlags;
///}

//// -----------------------------------------------------------------------
//// entry points from Matrix.cpp; calls into CPUMatrixTensorOpImpl
//// -----------------------------------------------------------------------

//// perform unary operation 'op' on a giving 'this', reinterpreting the matrices as tensors as specified by the dims and strides
//// This maps 'op' to a lambda.
///template <class ElemType>
///void CPUMatrix<ElemType>::TensorOp(ElemType beta, const CPUMatrix<ElemType>& a, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
///									const array<size_t, 2>& offsets,
///									const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
///									const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides)
///{
///	CPUMatrixTensorOpImpl<ElemType>(beta, a, *this, alpha, op, reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
///}

//// perform binary operation 'op' on a and b giving 'this', reinterpreting the matrices as tensors as specified by the dims and strides
//// This maps 'op' to a lambda.
///template <class ElemType>
///void CPUMatrix<ElemType>::TensorOp(ElemType beta, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
///									const array<size_t, 3>& offsets,
///									const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 3>& regularStrides,
///									const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 3>& reducingStrides)
///{
///	CPUMatrixTensorOpImpl<ElemType>(beta, a, b, *this, alpha, op, reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
///}

//// perform ternary operation 'op' on a, and c giving 'this', reinterpreting the matrices as tensors as specified by the dims and strides
//// This maps 'op' to a lambda.
///template <class ElemType>
///void CPUMatrix<ElemType>::TensorOp(ElemType beta, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, const CPUMatrix<ElemType>& c, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
///									const array<size_t, 4>& offsets,
///									const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 4>& regularStrides,
///									const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 4>& reducingStrides)
///{
///	CPUMatrixTensorOpImpl<ElemType>(beta, a, b, c, *this, alpha, op, reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
///}

///template <class ElemType>
///int CPUMatrix<ElemType>::Argmin() const
///{
///	int minArg = -1;
///	ElemType minValue = std::numeric_limits<ElemType>::max();

///#pragma omp parallel
///	{
///		int localMinArg = -1;
///		ElemType localMinValue = std::numeric_limits<ElemType>::max();

///		#pragma omp for
///		for (int index = 0; index < (int)GetNumElements(); ++index)
///		{
///			if (localMinValue > GetData()[index])
///			{
///				localMinArg = index;
///				localMinValue = GetData()[index];
///			}
///			// If we have more then one min value, select the one with lower index.
///			else if ((localMinValue == GetData()[index]) && (localMinArg > index))
///			{
///				localMinArg = index;
///			}
///		}

///		#pragma omp critical
///		{
///			if (minValue > localMinValue)
///			{
///				minArg = localMinArg;
///				minValue = localMinValue;
///			}
///			// If we have more then one min value, select the one with lower index.
///			else if ((minValue == localMinValue) && (minArg > localMinArg))
///			{
///				minArg = localMinArg;
///			}
///		}
///	}
///	return minArg;
///}

///template <class ElemType>
///int CPUMatrix<ElemType>::Argmax() const
///{
///	int maxArg = -1;
///	ElemType maxValue = std::numeric_limits<ElemType>::lowest();

///#pragma omp parallel
///	{
///		int localMaxArg = -1;
///		ElemType localMaxValue = std::numeric_limits<ElemType>::lowest();

///#pragma omp for
///		for (int index = 0; index < (int)GetNumElements(); ++index)
///		{
///			if (localMaxValue < GetData()[index])
///			{
///				localMaxArg = index;
///				localMaxValue = GetData()[index];
///			}
///			// If we have more then one max value, select the one with lower index.
///			else if ((localMaxValue == GetData()[index]) && (localMaxArg > index))
///			{
///				localMaxArg = index;
///			}
///		}

///#pragma omp critical
///		{
///			if (maxValue < localMaxValue)
///			{
///				maxArg = localMaxArg;
///				maxValue = localMaxValue;
///			}
///			// If we have more then one max value, select the one with lower index.
///			else if ((maxValue == localMaxValue) && (maxArg > localMaxArg))
///			{
///				maxArg = localMaxArg;
///			}
///		}
///	}
///	return maxArg;
///}

///template <class ElemType>
///int CPUMatrix<ElemType>::ArgOp(ElementWiseOperator reductionOp) const
///{
///	switch (reductionOp)
///	{
///		case ElementWiseOperator::opArgmin:
///			return Argmin();
///			break;
///		case ElementWiseOperator::opArgmax:
///			return Argmax();
///			break;
///	}

///	InvalidArgument("ArgOp: Arg reduction operations other than opArgmax, and opArgmin are not implemented");
///	return -1;
///}

///template <class ElemType>
///void CPUMatrix<ElemType>::TensorArgOp(const CPUMatrix<ElemType>& a, ElementWiseOperator reductionOp,
///										const array<size_t, 2>& offsets,
///										const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
///										const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides)
///{
///	CPUMatrixTensorArgOpImpl<ElemType>(a, *this, reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
///}

template <class ElemType>
void CPUMatrix<ElemType>::ScatterValues(ElemType* indices, ElemType* value, ElemType* data, ElemType alpha, size_t num_indices, size_t rows, size_t cols, size_t indices_step/*=1*/)
{
	ScatterValues(indices, value, data, alpha, num_indices, rows, cols, /*mask*/nullptr, /*numElemsPerMaskEntry*/0, indices_step);
}

template <class ElemType>
void CPUMatrix<ElemType>::ScatterValues(ElemType* indices, ElemType* value, ElemType* data, ElemType alpha, size_t num_indices, size_t rows, size_t cols, char* mask, size_t numElemsPerMaskEntry, size_t indices_step/*=1*/)
{
	if (!indices || !value || !data)
		LogicError("ScatterValues: input data is null");
	if (mask && (numElemsPerMaskEntry == 0))
		RuntimeError("ScatterValues: numElemsPerMaskEntry must not be 0 when a mask is provided");

#pragma omp parallel
	{
		//int ithread = omp_get_thread_num();
		//int nthread = omp_get_num_threads();
		for (auto i = 0; i < num_indices; i++)
		{
			auto col_r = indices[i * indices_step];
			if (std::isnan(col_r) || col_r < 0) continue;
			auto col = (size_t)col_r;
			//ignore the elements that is not partitioned into this thread
			//if (col % nthread != ithread) continue;
			//check if colMask is invalid
			if (mask && mask[i * indices_step / numElemsPerMaskEntry] == 0) continue;

			if (col >= cols)
				InvalidArgument("ScatterValues: Indices map out of bounds. %ld >= %ld", (long int)col, (long int)cols);

			auto index = col * rows;
			auto offset = i * rows;
			for (auto j = 0; j < rows; j++)
				data[index + j] = data[index + j] + alpha * value[offset + j];
		}
	}
}

///template <class ElemType>
///string CPUMatrix<ElemType>::str() const
///{
///	char sz[64];
///	sprintf_s(sz, sizeof(sz), "[%dx%d]", int(GetNumRows()), int(GetNumCols()));
///	string s(sz);
///	if (std::is_same<ElemType, float>::value) s += " float/cpu";
///	else if (std::is_same<ElemType, double>::value) s += " double/cpu";
///	else if (std::is_same<ElemType, half>::value) s += " half/cpu";
///	else s += " ***/cpu";
///	sprintf_s(sz, sizeof(sz), "  %04x", int(GetFormat())); s += sz;
///	return s;
///}

///template <class ElemType>
///void CPUMatrix<ElemType>::view(ostream& os) const
///{
///	os << str() << endl;
///	size_t nr = GetNumRows();
///	for (int j=0; j<nr; ++j)
///	{
///		const ElemType* p = GetData() + j;
///		for (int i=0; i<GetNumCols(); ++i) { os << "\t" << float(*p); p += nr; }
///		os << endl;
///	}
///}

//==============================================================================
//			CPUMatrix<half>
//==============================================================================

// General conversion function with no performance optimization
// this should only be used in CPU half precision
// For performance on inference on CPU, user should convert fp16 model to fp32 first, unless MKL supports half precision
///template<typename SrcT, typename DstT>
///static void ConvertBuffer(DstT* dst, const SrcT* src, size_t count)
///{
///	for (size_t i=0; i<count; ++i) *dst++ = (DstT)(*src++);
///}

// specialization to convert from half to float for computation, and then store in half
///template <>
///void CPUMatrix<half>::MultiplyAndWeightedAdd(half alpha, const CPUMatrix<half>& a, bool transposeA, const CPUMatrix<half>& b, bool transposeB,
///    half beta, CPUMatrix<half>& c, shared_ptr<QuantizedMultiplier<half>> pQuantizedMultiplier)
///{
///    CPUMatrix<float> af(a.GetNumRows(), a.GetNumCols());
///    CPUMatrix<float> bf(b.GetNumRows(), b.GetNumCols());
///    CPUMatrix<float> cf(c.GetNumRows(), c.GetNumCols());

///    if (alpha != 0)
///    {
///        ConvertBuffer<half, float>(af.GetData(), a.GetData(), a.GetNumElements());
///        ConvertBuffer<half, float>(bf.GetData(), b.GetData(), b.GetNumElements());
///    }
///    if (beta != 0)
///    {
///        ConvertBuffer<half, float>(cf.GetData(), c.GetData(), c.GetNumElements());
///    }
///    if (pQuantizedMultiplier)
///        RuntimeError("Quantized matrix multiply not supported for Half");

///    CPUMatrix<float>::MultiplyAndWeightedAdd((float)alpha, af, transposeA, bf, transposeB, (float)beta, cf, nullptr);
///	if (c.GetNumCols()!=cf.GetNumCols() || c.GetNumRows()!=cf.GetNumRows()) c.Resize(cf.GetNumRows(), cf.GetNumCols());
///    ConvertBuffer<float, half>(c.GetData(), cf.GetData(), c.GetNumElements());
///}

//// specialization to RunTimeError for now due to omp implementation only support build-in type
///template <>
///void CPUMatrix<half>::AssignSoftmaxSum(const CPUMatrix<half>& softmax, CPUMatrix<half>& c)
///{
///    RuntimeError("half AssignSoftmaxSum not supported.");
///}

///template <>
///void CPUMatrix<half>::AssignNCEUnnormalizedEval(const CPUMatrix<half>& a,
///                                                const CPUMatrix<half>& b, const CPUMatrix<half>& bias, CPUMatrix<half>& c)
///{
///    RuntimeError("half AssignNCEUnnormalizedEval not supported.");
///}

///template <>
///void CPUMatrix<half>::VectorSum(const CPUMatrix<half>& a, CPUMatrix<half>& c, bool isColWise)
///{
///    RuntimeError("half VectorSum not supported.");
///}

///template <>
///void CPUMatrix<half>::VectorNorm1(CPUMatrix<half>& c, bool isColWise) const
///{
///    RuntimeError("half VectorNorm1 not supported.");
///}

///template <>
///half CPUMatrix<half>::SumOfElements() const
///{
///    float acc = 0;
///	half* p = GetData();
///	size_t len = GetNumRows()*GetNumCols();
///	for (size_t j=0; j<len; ++j) acc += float(*p++);
///	return half(acc);
///}

///template <>
///half CPUMatrix<half>::MatrixNorm1() const
///{
///    RuntimeError("half MatrixNorm1 not supported.");
///}

///template <>
///    half CPUMatrix<half>::FrobeniusNorm() const
///{
///    RuntimeError("half FrobeniusNorm not supported.");
///}

///template <>
///void CPUMatrix<half>::MaxPoolingBackward(const CPUMatrix<half>& out, const CPUMatrix<half>& in,
///                                         const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIndices, const CPUMatrix<int>& indices,
///                                         CPUMatrix<half>& grad, bool accumulateGradient) const
///{
///    RuntimeError("half MaxPoolingBackward not supported.");
///}

///template <>
///void CPUMatrix<half>::MaxROIPoolingBackward(const size_t numRois, const size_t numImg, const size_t channels, const size_t width, const size_t height,
///                                            const size_t pooledWidth, const size_t pooledHeight, const CPUMatrix<half>& roiData, CPUMatrix<half>& grad,
///                                            CPUMatrix<half>& argmax, double spatialScale) const
///{
///    RuntimeError("half MaxROIPoolingBackward not supported.");
///}

///template <>
///void CPUMatrix<half>::AveragePoolingBackward(const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIndices, const CPUMatrix<int>& indices, CPUMatrix<half>& grad, bool poolIncludePad, bool accumulateGradient) const
///{
///    RuntimeError("half AveragePoolingBackward not supported.");
///}

// instantiate templated methods
///template void CPUMatrix<float>::AdaDelta(CPUMatrix<float>& gradients, CPUMatrix<float>& functionValues, float learningRate, float rho, float epsilon);
///template void CPUMatrix<double>::AdaDelta(CPUMatrix<double>& gradients, CPUMatrix<double>& functionValues, double learningRate, double rho, double epsilon);
///template void CPUMatrix<float>::AdaDelta(CPUMatrix<half>& gradients, CPUMatrix<float>& functionValues, float learningRate, float rho, float epsilon);

///template void CPUMatrix<float>::BatchNormalizationForward(const CPUMatrix<float>& scale, const CPUMatrix<float>& bias, bool inferenceOnly, double expAvgFactor, double blendFactor, CPUMatrix<float>& runMean, CPUMatrix<float>& runVariance, CPUMatrix<float>& out, double epsilon, CPUMatrix<float>& saveMean, CPUMatrix<float>& saveInvStdDev) const;
///template void CPUMatrix<double>::BatchNormalizationForward(const CPUMatrix<double>& scale, const CPUMatrix<double>& bias, bool inferenceOnly, double expAvgFactor, double blendFactor, CPUMatrix<double>& runMean, CPUMatrix<double>& runVariance, CPUMatrix<double>& out, double epsilon, CPUMatrix<double>& saveMean, CPUMatrix<double>& saveInvStdDev) const;
///template void CPUMatrix<half>::BatchNormalizationForward(const CPUMatrix<float>& scale, const CPUMatrix<float>& bias, bool inferenceOnly, double expAvgFactor, double blendFactor, CPUMatrix<float>& runMean, CPUMatrix<float>& runVariance, CPUMatrix<half>& out, double epsilon, CPUMatrix<float>& saveMean, CPUMatrix<float>& saveInvStdDev) const;

///template void CPUMatrix<float>::BatchNormalizationBackward(const CPUMatrix<float>& in, CPUMatrix<float>& grad, const CPUMatrix<float>& scale, double blendFactor, const CPUMatrix<float>& saveMean, const CPUMatrix<float>& saveInvStdDev, CPUMatrix<float>& scaleGrad, CPUMatrix<float>& biasGrad) const;
///template void CPUMatrix<double>::BatchNormalizationBackward(const CPUMatrix<double>& in, CPUMatrix<double>& grad, const CPUMatrix<double>& scale, double blendFactor, const CPUMatrix<double>& saveMean, const CPUMatrix<double>& saveInvStdDev, CPUMatrix<double>& scaleGrad, CPUMatrix<double>& biasGrad) const;
///template void CPUMatrix<half>::BatchNormalizationBackward(const CPUMatrix<half>& in, CPUMatrix<half>& grad, const CPUMatrix<float>& scale, double blendFactor, const CPUMatrix<float>& saveMean, const CPUMatrix<float>& saveInvStdDev, CPUMatrix<float>& scaleGrad, CPUMatrix<float>& biasGrad) const;

//==============================================================================
//			CPUMatrix<half>
//==============================================================================

template class CPUMatrix<half>;

//int CPUMatrix<half>::m_optimizationFlags = 0;

//==============================================================================
//			CPUMatrix<float>
//==============================================================================

template class CPUMatrix<float>;

//int CPUMatrix<float>::m_optimizationFlags = CPUMatrix<float>::OPT_EVAL_WITH_MKL; // enable eval MKL optimization by default

//==============================================================================
//			CPUMatrix<double>
//==============================================================================

template class CPUMatrix<double>;

//int CPUMatrix<double>::m_optimizationFlags = 0;

//==============================================================================
//			CPUMatrix<char>
//==============================================================================

// We use Matrix<char> as the backing store for QuantizedMatrix
// Let's explicitly instantiate the methods we need for that purpose
///template CPUMatrix<char>::CPUMatrix();
template CPUMatrix<char>::CPUMatrix(int flags);
template CPUMatrix<char>::CPUMatrix(size_t rows, size_t cols, int flags);
template CPUMatrix<char>::CPUMatrix(size_t rows, size_t cols, char* p, int flags);
template CPUMatrix<char>::CPUMatrix(CPUMatrix<char> const&);
template CPUMatrix<char>::CPUMatrix(CPUMatrix<char>&&);
///template size_t CPUMatrix<char>::ItemPos(size_t, size_t) const;
///template CPUMatrix<char> CPUMatrix<char>::GetColumnSlice(size_t startColumn, size_t cols) const;
///template CPUMatrix<char>& CPUMatrix<char>::operator=(CPUMatrix<char>&&);
template char& CPUMatrix<char>::operator()(size_t row, size_t col);
template const char& CPUMatrix<char>::operator()(size_t row, size_t col) const;
///template void CPUMatrix<char>::SetValue(const char);
///template void CPUMatrix<char>::SetValue(size_t rows, size_t cols, char* pArray, int flags);
///template void CPUMatrix<char>::SetValue(CPUMatrix<char> const&);
///template bool CPUMatrix<char>::IsEqualTo(const CPUMatrix<char>& a, const char threshold) const;
//template void CPUMatrix<char>::SetValue(GPUMatrix<char> const&);
//template void CPUMatrix<char>::SetValue(CPUSparseMatrix<char> const&);
//template void CPUMatrix<char>::SetValue(GPUSparseMatrix<char> const&);
//template void CPUMatrix<char>::Resize(size_t rows, size_t cols, bool growOnly);
//template void CPUMatrix<char>::Resize(size_t rows, size_t cols, bool growOnly);
///template char* CPUMatrix<char>::CopyToArray() const;
///template void CPUMatrix<char>::CopySection(size_t rows, size_t cols, char* dst, size_t colStride) const;
///template void CPUMatrix<char>::Reshape(size_t, size_t);
///template void CPUMatrix<char>::SetUniformRandomValue(const char low, const char high, unsigned long seed);
///template void CPUMatrix<char>::SetUniformRandomValue(RNGHandle& rngHandle, const char low, const char high);
///template void CPUMatrix<char>::SetGaussianRandomValue(const char mean, const char sigma, unsigned long seed);

//==============================================================================
//			CPUMatrix<short>
//==============================================================================

///template CPUMatrix<short>::CPUMatrix();
template CPUMatrix<short>::CPUMatrix(int flags);
template CPUMatrix<short>::CPUMatrix(size_t rows, size_t cols, int flags);
template CPUMatrix<short>::CPUMatrix(size_t rows, size_t cols, short* p, int flags);
template CPUMatrix<short>::CPUMatrix(CPUMatrix<short> const&);
template CPUMatrix<short>::CPUMatrix(CPUMatrix<short>&&);
///template size_t CPUMatrix<short>::ItemPos(size_t, size_t) const;
///template CPUMatrix<short> CPUMatrix<short>::GetColumnSlice(size_t startColumn, size_t cols) const;
///template CPUMatrix<short>& CPUMatrix<short>::operator=(CPUMatrix<short>&&);
template short& CPUMatrix<short>::operator()(size_t row, size_t col);
template const short& CPUMatrix<short>::operator()(size_t row, size_t col) const;
///template void CPUMatrix<short>::SetValue(const short);
///template void CPUMatrix<short>::SetValue(size_t rows, size_t cols, short* pArray, int flags);
///template void CPUMatrix<short>::SetValue(CPUMatrix<short> const&);
///template bool CPUMatrix<short>::IsEqualTo(const CPUMatrix<short>& a, const short threshold) const;
//template void CPUMatrix<short>::SetValue(GPUMatrix<short> const&);
//template void CPUMatrix<short>::SetValue(CPUSparseMatrix<short> const&);
//template void CPUMatrix<short>::SetValue(GPUSparseMatrix<short> const&);
//template void CPUMatrix<short>::Resize(size_t rows, size_t cols, bool growOnly);
//template void CPUMatrix<short>::Resize(size_t rows, size_t cols, bool growOnly);
///template short* CPUMatrix<short>::CopyToArray() const;
///template void CPUMatrix<short>::CopySection(size_t rows, size_t cols, short* dst, size_t colStride) const;
///template void CPUMatrix<short>::Reshape(size_t, size_t);
///template void CPUMatrix<short>::SetUniformRandomValue(const short low, const short high, unsigned long seed);
///template void CPUMatrix<short>::SetUniformRandomValue(RNGHandle& rngHandle, const short low, const short high);
///template void CPUMatrix<short>::SetGaussianRandomValue(const short mean, const short sigma, unsigned long seed);

//==============================================================================
//			CPUMatrix<int>
//==============================================================================

///template CPUMatrix<int>::CPUMatrix(size_t, size_t, int*, int);

} } }
