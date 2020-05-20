//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "../Common/Basics.h"
#include "../Common/File.h"
#include "CPUSparseMatrix.h"
#include "CPUMatrix.h"
//#include "half.hpp"
//#include <assert.h>
//#include <stdexcept>
//#include <omp.h>
//#include <math.h>
//#include <random>
//#include <chrono>
//#include <iostream>

//#pragma warning(disable : 4127) // conditional expression is constant; "if (sizeof(ElemType)==sizeof(float))" triggers this

#ifdef USE_MKL
// requires MKLML 0.11 and above
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#else
#ifdef _MSC_VER
// Visual Studio doesn't define standard complex types properly
#define HAVE_LAPACK_CONFIG_H
#define LAPACK_COMPLEX_STRUCTURE
#endif
#include <cblas.h>
#include <lapacke.h>
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

///template <class ElemType>
///CPUSparseMatrix<ElemType>::CPUSparseMatrix(MatrixFormat mft, size_t rows, size_t cols, size_t n) : Base(mft, CPUDEVICE)
///{
///	Resize(rows, cols);
///	if (n) Allocate(n);
///}
///
////-------------------------------------------------------------------------
//// construction and conversion
////-------------------------------------------------------------------------
///
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::SetDiagonalValue(ElemType v)
///{
///	Reset();
///	MatrixFormat mft = GetFormat();
///	size_t nc = (mft & matrixFormatRowMajor) ? m_numRows : m_numCols;
///	size_t nr = (mft & matrixFormatRowMajor) ? m_numCols : m_numRows;
///	size_t n = GetDiagSize();
///
///	if (mft & matrixFormatBlock)
///	{
///		// block
///		Allocate(n*nr);
///		ElemType* p = Buffer(); memset(p, 0, n*nr*sizeof(ElemType));
///		size_t* blockPos = GetBlockPos();
///		for (size_t j=0; j<n; ++j) { blockPos[j] = j; p[j] = v; p += nr; }
///		SetBlockCount(n);
///	}
///	else
///	{
///		// sparse
///		index_t* compPos = GetCompPos();
///		index_t* compId = GetCompId();
///		ElemType* p = Buffer();
///		for (size_t j=0; j<n; ++j) { compPos[j] = compId[j] = index_t(j); *p++ = v; }
///		for (size_t j=n; j<=nc; ++j) compPos[j] = index_t(n);
///	}
///}
///
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::SetDiagonalValue(const CPUMatrix<ElemType>& v)
///{
///	if (v.GetNumRows() != 1 && v.GetNumCols() != 1)
///		LogicError("SetDiagonalValue: input vector must be a vector");
///
///	size_t dsize = v.GetNumElements();
///	if (dsize == 1) { SetDiagonalValue(v(0,0)); return; }
///	if (dsize != GetDiagSize())
///		LogicError("SetDiagonalValue: input vector's dimension does not agree with [this]");
///
///	Reset();
///	MatrixFormat mft = GetFormat();
///	if (mft & matrixFormatBlock)
///	{
///		// block
///		size_t nr = (mft & matrixFormatRowMajor) ? m_numCols : m_numRows;
///		Allocate(dsize*nr);
///		const ElemType* src = v.Data();
///		ElemType* dst = Data();
///
///		size_t* blockPos = GetBlockPos();
///		for (size_t j=0; j<dsize; ++j) { blockPos[j] = j; dst[j] = *src++; dst += nr; }
///		SetBlockCount(dsize);
///	}
///	else
///	{
///		// sparse
///		Allocate(dsize);
///		const ElemType* src = v.Data();
///		ElemType* dst = Data();
///
///		index_t* compPos = GetCompPos();
///		index_t* compId = GetCompId();
///		size_t nc = (mft & matrixFormatRowMajor) ? m_numRows : m_numCols;
///		for (size_t j=0; j<dsize; ++j) { compPos[j] = compId[j] = (index_t)j; *dst++ = *src++; }
///		for (size_t j=dsize; j<=nc; ++j) compPos[j] = (index_t)dsize;
///	}
///}
///
///template <class ElemType>
///CPUSparseMatrix<ElemType>& CPUSparseMatrix<ElemType>::AssignOneHot(const CPUMatrix<ElemType>& a, vector<size_t>& shape, size_t axis)
///{
///	if (a.IsEmpty()) LogicError("AssignOneHot: Matrix a is empty");
///	if (GetFormat() != matrixFormatSparseCSC) LogicError("AssignOneHot: Matrix format is not supported");
///	if (axis >= shape.size()) LogicError("AssignOneHot: axis is not correct");
///
///	int itemSize = 1;
///	for (size_t i=0; i<axis; ++i) itemSize *= (int)shape[i];
///	int numClass = (int)shape[axis];
///
///	size_t nSize = a.GetNumElements();
///	size_t nRows = itemSize * numClass;
///	size_t nCols = nSize / itemSize;
///	//if (m_numRows==0 || m_numRows!=nRows || m_numCols==0 || m_numCols!=nCols)
///	//	LogicError("AssignOneHot: Target matrix size is not correct");
///
///	Reset();
///	Resize(nRows, nCols);
///	Allocate(nSize);
///
///	index_t* compPos = GetCompPos();
///	index_t* compId = GetCompId();
///	const ElemType* src = a.Data();
///	ElemType* dst = Data();
///#pragma omp parallel for
///	for (long j=0; j<nSize; ++j)
///	{
///		int blockId = j / itemSize;	// cid
///		int itemId = j % itemSize;	// rid
///		// for invalid indices, theorically they should not belong to nz elements.
///		// but if we scan the indices to count the valid indices number,
///		// it will be difficult for parallel calculation, especially on GPU.
///		// here we chose to keep those elements in nz element list, but with value 0 an at row 0
///		if (src[j]<0 || src[j]>=numClass) { compId[j] = index_t(itemId); dst[j] = 0; }
///		else { compId[j] = int(src[j])*itemSize + itemId; dst[j] = 1; }
///		if (itemId == 0) compPos[blockId+1] = index_t(itemSize*(blockId+1));
///	}
///	compPos[0] = 0;
///
///	return *this;
///}
///
///#pragma endregion Constructors and Destructor
///
///#pragma region Basic Operators
///
/// make sure call order in column wise for CSC and row wise for CSR
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::SetValue(size_t row, size_t col, ElemType v)
///{
///	if (!OwnBuffer())
///		LogicError("Cannot modify since the buffer is managed externally");
///	if (GetFormat() != matrixFormatSparseCSC && GetFormat() != matrixFormatSparseCSR)
///		LogicError("CPUSparseMatrix:  unsupported SetValue() call");
///
///	if ((GetFormat() == matrixFormatSparseCSC) && ((*this)(row, col) == v))
///		return;
///
///	let nz = NzCount();
///	if (GetSizeAllocated() < nz + 1) // automatic resize
///		Allocate(m_numRows, m_numCols, nz + 100, true, true); // allocate 100 more elelemnts and keep existing values
///
///	if (row < 0 || row >= m_numRows) LogicError("CPUSparseMatrix: SetValue() invalid row id");
///	if (col < 0 || col >= m_numCols) LogicError("CPUSparseMatrix: SetValue() invalid column id");
///
///	size_t r = (GetFormat() == matrixFormatSparseCSC) ? row : col;
///	size_t c = (GetFormat() == matrixFormatSparseCSC) ? col : row;
///
///	Data()[nz] = v;
///	MajorIndexLocation()[nz] = (index_t) r;
///
///	// consistency check
///	if (nz > 0)
///	{
///		if (c == GetColIdx() && r <= MajorIndexLocation()[nz - 1])
///			LogicError("CPUSparseMatrix:  SetValue is not called properly");
///	}
///	if (c != GetColIdx())
///	{
///		SecondIndexLocation()[c] = index_t(nz);
///		SetColIdx((int) c);
///	}
///	// Note we don't have m_nz anymore. In order for the change from m_nz to
///	// NzCount to make sense, we need to propogate nz+1 to all col slices.
///	for (size_t max = c + 1; max < m_numCols + 1; max++)
///		SecondIndexLocation()[max] = index_t(nz + 1);
///}

// make sure call order in colume wise for CSC and row wise for CSR
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::SetValue(const CPUSparseMatrix<ElemType>& mat)
///{
///	if (&mat == this) return;
///
///	MatrixFormat mft = mat.GetFormat();
///
///	Init(mft, CPUDEVICE);
///	Resize(mat.GetNumRows(), mat.GetNumCols()); if (IsEmpty()) return;
///	size_t nc = (mft & matrixFormatRowMajor) ? m_numRows : m_numCols;
///	size_t nr = (mft & matrixFormatRowMajor) ? m_numCols : m_numRows;
///
///	if (mft == matrixFormatSparseCSC || mft == matrixFormatSparseCSR)
///	{
///		index_t* primePos = mat.GetPrimePos();
///		size_t ns = primePos[0], ne = primePos[nc];
///		size_t n = ne - ns; if (n==0) { Allocate(max(nc,nr)); return; }
///
///		Allocate(n);
///		index_t* compPos = GetCompPos();
///		memcpy(compPos, primePos, (nc+1)*sizeof(index_t));
///		for (size_t j=1; j<=nc; ++j) compPos[j] -= compPos[0]; compPos[0] = 0;
///		memcpy(GetCompId(), mat.GetCompId()+ns, n*sizeof(index_t));
///		memcpy(Buffer(), mat.Buffer()+ns, n*sizeof(ElemType));
///	}
///	else if (mft == matrixFormatSparseBSC || mft == matrixFormatSparseBSR)
///	{
///		size_t n = 0;
///		size_t* srcPos = mat.GetBlockPos();
///		size_t* dstPos = GetBlockPos();
///		for (size_t j=0; j<nc; ++j) if (srcPos[j]!=string::npos) ++n;
///
///		Allocate(n==0 ? nr : n*nr); n = 0;
///		const ElemType* src = mat.Buffer();
///		ElemType* dst = Buffer();
///		for (size_t j=0; j<nc; ++j)
///		{
///			size_t i = srcPos[j];
///			if (i==string::npos) { dstPos[j] = i; continue; }
///			memcpy(dst+n*nr, src+srcPos[j]*nr, nr*sizeof(ElemType));
///			dstPos[j] = n++;
///		}
///		SetBlockCount(n);
///	}
///}

///#if 0
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::SetValue(const CPUMatrix<ElemType>& /*v*/)
///{
///	NOT_IMPLEMENTED;
///}
///
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::SetValue(const GPUMatrix<ElemType>& /*v*/)
///{
///	NOT_IMPLEMENTED;
///}
///
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::SetValue(const GPUSparseMatrix<ElemType>& /*v*/)
///{
///	NOT_IMPLEMENTED;
///}
///#endif

///template <class ElemType>
///void CPUSparseMatrix<ElemType>::MaskColumnsValue(const CPUMatrix<char>& mask, ElemType val, size_t mcols)
///{
///	NOT_IMPLEMENTED;
///}
///
//// this[:,j] = a[:,idx[j]] * alpha + this[:,j] * beta
///template <class ElemType>
///CPUSparseMatrix<ElemType>& CPUSparseMatrix<ElemType>::DoGatherColumnsOf(ElemType beta, const CPUMatrix<ElemType>& idx, const CPUSparseMatrix<ElemType>& a, ElemType alpha)
///{
///	VerifyWritable(__func__);
///
///	if ((a.GetFormat() != matrixFormatSparseCSC) || (GetFormat() != matrixFormatSparseCSC))
///		NOT_IMPLEMENTED;
///
///	if (idx.GetNumRows() != 1) // index is 1-dimensional only
///		InvalidArgument("DoGatherColumnsOf: Map must be a row vector");
///
///	if (beta != 0)
///		NOT_IMPLEMENTED;
///
///	// Determine the number of non-zero elements
///	size_t numCols = idx.GetNumCols();
///	size_t numNonZeroElements = 0;
///	// TODO: Does it make sense to parallelize this?
///	for (long j = 0; j < numCols; j++)
///	{
///		auto jInF = idx(0, j); // this is the column we need to get
///		if (std::isnan(jInF) || (jInF < 0))     // negative index means gap
///			continue;
///		size_t jIn = (size_t)jInF;
///
///		auto start = a.SecondIndexLocation()[jIn];
///		auto end = a.SecondIndexLocation()[jIn + 1];
///		numNonZeroElements += (end - start);
///	}
///
///	if (beta == 0)
///		RequireSizeAndAllocate(a.GetNumRows(), idx.GetNumCols(), numNonZeroElements); // output has same column format as a, but number of columns comes from idx
///
///	size_t offset = SecondIndexLocation()[0];
///	// TODO: Does it make sense to parallelize this?
///	for (long j = 0; j < numCols; j++)
///	{
///		auto jInF = idx(0, j); // this is the column we need to get
///		if (jInF >= 0)     // negative or nan index means gap, but we still need to update the CompIndex
///		{
///			size_t jIn = (size_t)jInF;
///
///			auto start = a.SecondIndexLocation()[jIn];
///			auto end = a.SecondIndexLocation()[jIn + 1];
///			for (auto p = start; p < end; p++, offset++)
///			{
///				GetCompId()[offset] = a.GetCompId()[p];
///				Buffer()[offset] = a.Buffer()[p] * alpha;
///			}
///		}
///		SecondIndexLocation()[j + 1] = index_t(offset);
///	}
///	return *this;
///}
///
//// *this[:,idx[j]] = a[:,j] * alpha + *this[:,idx[j]] * beta
///template <class ElemType>
///CPUSparseMatrix<ElemType>& CPUSparseMatrix<ElemType>::DoScatterColumnsOf(ElemType beta, const CPUMatrix<ElemType>& idx, const CPUSparseMatrix<ElemType>& a, ElemType alpha)
///{
///	VerifyWritable(__func__);
///
///	if ((a.GetFormat() != matrixFormatSparseCSC) || (GetFormat() != matrixFormatSparseCSC))
///		NOT_IMPLEMENTED;
///
///	if (idx.GetNumRows() != 1) // index is 1-dimensional only
///		InvalidArgument("DoScatterColumnsOf: Map must be a row vector");
///
///	if (beta != 0)
///		NOT_IMPLEMENTED;
///
///	if (NzCount() != 0)
///		InvalidArgument("CPUSparseMatrix::DoScatterColumnsOf: The target matrix cannot have pre-existing non-zero values when being scattered into");
///
///	size_t numNonZeroElements = a.NzCount();
///
///	if (beta == 0)
///		RequireSizeAndAllocate(GetNumRows(), GetNumCols(), numNonZeroElements);
///
///	// Setup the Secondary index
///	std::vector<int> columnElementCounts(GetNumCols(), 0);
///	size_t numColsToWrite = idx.GetNumCols();
///	for (long j = 0; j < numColsToWrite; j++)
///	{
///		auto jOutF = idx(0, j); // this is the column we need to write to
///		if (std::isnan(jOutF) || (jOutF < 0)) continue;	// negative index means gap
///		size_t jOut = (size_t)jOutF;
///		columnElementCounts[jOut] = a.SecondIndexLocation()[j + 1] - a.SecondIndexLocation()[j];
///	}
///
///	// TODO: Replace with std::exclusive_scan when we switch to C++17
///	for (size_t i = 1; i <= GetNumCols(); ++i)
///		SecondIndexLocation()[i] = SecondIndexLocation()[i - 1] + columnElementCounts[i - 1];
///
///	size_t offset = a.SecondIndexLocation()[0];
///	// TODO: Does it make sense to parallelize this?
///	for (long j = 0; j < numColsToWrite; j++)
///	{
///		auto jOutF = idx(0, j); // this is the column we need to write to
///		if (std::isnan(jOutF) || (jOutF < 0))     // negative index means gap
///			continue;
///		size_t jOut = (size_t)jOutF;
///
///		auto start = SecondIndexLocation()[jOut];
///		auto end = SecondIndexLocation()[jOut + 1];
///		for (auto p = start; p < end; p++, offset++)
///		{
///			GetCompId()[p] = a.GetCompId()[offset];
///			Buffer()[p] = a.Buffer()[offset] * alpha;
///		}
///	}
///	return *this;
///}

///template <class ElemType>
///void CPUSparseMatrix<ElemType>::Print(const char* matrixName) const
///{
///	Print(matrixName, 0, 0, 0, 0);
///}
///
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::Print(const char* matrixName, ptrdiff_t /*rowStart*/, ptrdiff_t /*rowEnd*/, ptrdiff_t /*colStart*/, ptrdiff_t /*colEnd*/) const
///{
///	if (GetFormat() != matrixFormatSparseCSC && GetFormat() != matrixFormatSparseCSR)
///		// NOT_IMPLEMENTED;
///		return;
///
///	fprintf(stderr, "%s\n", matrixName);
///
///	const ElemType* dataBuffer = NzValues();
///	const size_t nz = MajorIndexCount();
///	index_t* unCompressedIndex = MajorIndexLocation();
///	index_t* compressedIndex = SecondIndexLocation();
///
///	for (size_t i = 0, j = 0; i < nz; ++i)
///	{
///		if (i >= compressedIndex[j]) { fprintf(stderr, "\n"); j++; }
///		fprintf(stderr, "%d:%.f ", unCompressedIndex[i], (double)dataBuffer[i]);
///	}
///	fprintf(stderr, "\n");
///}
///
///template <class ElemType>
///CPUSparseMatrix<ElemType> CPUSparseMatrix<ElemType>::GetColumnSlice(size_t start, size_t len) const
///{
///	if (start + len > m_numCols)
///		InvalidArgument("The slice (%d+%d) is out of range of the source matrix (%d).", (int)start, (int)len, (int)m_numCols);
///
///	if (GetFormat() != matrixFormatSparseCSC && GetFormat() != matrixFormatSparseBSC)
///		NOT_IMPLEMENTED;
///
///	CPUSparseMatrix<ElemType> slice(GetFormat());
///	slice.Assign(*this, true);
///	slice.SetSlice(start, len);
///	return slice;
///}
///
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::AssignColumnSliceToDense(CPUMatrix<ElemType>& slice, size_t start, size_t len) const
///{
///	if (start+len > m_numCols)
///		InvalidArgument("The slice (%d+%d) is out of range of the source matrix (%d).", (int)start, (int)len, (int)m_numCols);
///
///	slice.Resize(m_numRows, len);
///	memset(slice.Data(), 0, m_numRows*len*sizeof(ElemType));
///
///	MatrixFormat mft = GetFormat();
///	if (mft == matrixFormatSparseCSC)
///	{
///		ElemType* src = Buffer();
///		ElemType* dst = slice.Buffer();
///		index_t* primePos = GetPrimePos() + start;
///		index_t* compId = GetCompId();
///#pragma omp parallel for
///		for (long j=0; j<len; ++j)
///		{
///			long ns = primePos[j], ne = primePos[j+1];
///			for (long k=ns; k<ne; ++k) dst[compId[k]] = src[k];
///			dst += m_numRows;
///		}
///	}
///	else if (mft == matrixFormatSparseBSC)
///	{
///		ElemType* src = Buffer();
///		ElemType* dst = slice.Buffer();
///		size_t* blockPos = GetBlockPos() + start;
///#pragma omp parallel for
///		for (long j=0; j<len; ++j)
///		{
///			size_t k = blockPos[j];
///			if (k!=string::npos) memcpy(dst, src+k*m_numRows, m_numRows*sizeof(ElemType));
///			dst += m_numRows;
///		}
///	}
///	else NOT_IMPLEMENTED
///}
///
///template <class ElemType>
///CPUMatrix<ElemType> CPUSparseMatrix<ElemType>::CopyColumnSliceToDense(size_t start, size_t len) const
///{
///	CPUMatrix<ElemType> slice(m_numRows, len);
///	AssignColumnSliceToDense(slice, start, len);
///	return slice;
///}
///
///template <class ElemType>
///CPUMatrix<ElemType> CPUSparseMatrix<ElemType>::DiagonalToDense() const
///{
///	if (m_numRows != m_numCols)
///		LogicError("DiagonalToDense can be called only for square matrix");
///
///	CPUMatrix<ElemType> diag(1, m_numCols);
///	MatrixFormat mft = GetFormat();
///	if ((mft & matrixFormatBlock)==0)
///	{
///		// sparse
///		const index_t* primePos = GetPrimePos();
///		const index_t* compId = GetCompId();
///		const ElemType* src = Buffer();
///		ElemType* dst = diag.Buffer();
///		for (size_t j=0; j<m_numCols; ++j)
///		{
///			size_t ns = primePos[j], ne = primePos[j+1];
///			for (size_t i=ns; i<ne; ++i)
///				if (compId[i]==j) { dst[j] = src[i]; break; }
///		}
///	}
///	else
///	{
///		// block
///		const size_t* blockPos = GetBlockPos();
///		const ElemType* src = Buffer();
///		ElemType* dst = diag.Buffer();
///		for (size_t j=0; j<m_numCols; ++j)
///		{
///			size_t k = blockPos[j];
///			if (k!=string::npos) dst[j] = src[k*m_numRows+j];
///		}
///	}
///	return diag;
///}
///
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::SetMatrixFromCSCFormat(const index_t* pCol, const index_t* pRow, const ElemType* pVal, const size_t nz, size_t numRows, size_t numCols)
///{
///	if (!OwnBuffer())
///		LogicError("Cannot modify since the buffer is managed externally");
///
///	SetFormat(matrixFormatSparseCSC);
///	RequireSizeAndAllocate(numRows, numCols, nz, true, false);
///	memcpy(ColLocation(), h_CSCCol, sizeof(index_t)*(numCols + 1));
///	memcpy(RowLocation(), h_Row, sizeof(index_t)*nz);
///	memcpy(NzValues(), h_Val, sizeof(ElemType)*nz);
///
///	Init(matrixFormatSparseCSC,CPUDEVICE);
///	Resize(numRows, numCols);
///	Allocate(nz);
///
///	index_t* compPos = GetCompPos();
///	memcpy(compPos, pCol, (numCols+1)*sizeof(index_t));
///	//if (compPos[0]) { for (size_t j=1; j<=numCols; ++j) compPos[j] -= compPos[0]; compPos[0] = 0; }
///	memcpy(GetCompId(), pRow, nz*sizeof(index_t));
///	memcpy(Buffer(), pVal, nz*sizeof(ElemType));
///}
///
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::SetMatrixFromBSCFormat(const size_t* blockPos, const ElemType* pVal, size_t numBlocks, size_t numRows, size_t numCols)
///{
///	if (!OwnBuffer())
///		LogicError("Cannot modify since the buffer is managed externally");
///
///	SetFormat(matrixFormatSparseBSC);
///	Resize(numRows, numCols, numBlocks * numRows);
///	SetBlockSize(numBlocks);
///
///	memcpy(GetBlockIds(), blockIds, sizeof(size_t)*(numBlocks));
///	memcpy(Data(), val, sizeof(ElemType)*numBlocks*numRows);
///
///	Init(matrixFormatSparseBSC,CPUDEVICE);
///	Resize(numRows, numCols);
///	Allocate(numBlocks*numRows);
///
///	memcpy(GetBlockPos(), blockPos, numCols*sizeof(size_t));
///	memcpy(Buffer(), pVal, numBlocks*numRows*sizeof(ElemType));
///	SetBlockCount(numBlocks);
///}
///
///template <class ElemType>
///ElemType* CPUSparseMatrix<ElemType>::Data()  const
///{
///	MatrixFormat mft = GetFormat();
///	return (Buffer() + (mft == matrixFormatSparseCSC || mft == matrixFormatSparseCSR ? GetCompPos()[m_sliceOffset] : 0));
///}
///
//// WARNING: When memory is reallocated, existing information will be lost.
//// TODO: add keepval (default to true) argument so that the existing values are kept even after reallocation
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::Allocate(size_t numRows, size_t numCols, size_t numNZElemRequested, bool growOnly, bool keepval)
///{
///	if (m_numRows != numRows || m_numCols != numCols)
///		LogicError("Error, calling allocate with dimensions (%d, %d), but the matrix has dimension (%d, %d).", (int)numRows, (int)numCols, (int)GetNumRows(), (int)GetNumCols());
///
///	size_t numNzElem = max(numNZElemRequested, 1);
///	size_t newCompIndexSize;
///	if (GetFormat() == matrixFormatSparseCSC) newCompIndexSize = numCols + 1;
///	else if (GetFormat() == matrixFormatSparseCSR) newCompIndexSize = numRows + 1;
///	else newCompIndexSize = (numCols > numRows ? numCols : numRows) + 1;
///
///	bool reallocate = (GetSizeAllocated() < numNzElem || 
///					(GetSizeAllocated() > numNzElem && !growOnly) || 
///					GetCompPosSize() < newCompIndexSize);
///	if (reallocate)
///	{
///		if (GetFormat() == matrixFormatSparseCSC || 
///			GetFormat() == matrixFormatSparseCSR)
///		{
///			// The initialization of the following buffer is done by new []().
///			auto* pArray      = new ElemType[numNzElem]();
///			auto* unCompIndex = new index_t[numNzElem]();
///			auto* compIndex   = new index_t[newCompIndexSize]();
///
///			if (keepval && (NzCount() > numNzElem || GetCompPosSize() > newCompIndexSize))
///				LogicError("Allocate: To keep values m_nz should <= numNzElem and m_compIndexSize <= newCompIndexSize");
///
///			if (keepval && NzCount() > 0)
///			{
///				assert(GetCompPosSize() > 0 && NzCount() < numNzElem);
///				memcpy(pArray, Data(), NzSize());
///				memcpy(unCompIndex, GetCompId(), MajorIndexSize());
///				memcpy(compIndex, GetCompPos(), SecondIndexSize());
///			}
///			// TODO: This is super ugly. The internals of the storage object should be a shared_ptr.
///			delete[] Buffer();
///			delete[] GetCompId();
///			delete[] GetCompPos();
///
///			SetBuffer(pArray, numNzElem, false);
///			SetCompId(unCompIndex);
///			SetCompPos(compIndex);
///		}
///		else if (GetFormat() == matrixFormatSparseBSC || 
///				GetFormat() == matrixFormatSparseBSR)
///		{
///			ElemType* blockVal = new ElemType[numNzElem];
///			size_t* blockIds = new size_t[newCompIndexSize];
///
///			if (keepval && (NzCount() > numNzElem || GetCompPosSize() > newCompIndexSize))
///				LogicError("Resize: To keep values m_nz should <= numNzElem and m_compIndexSize <= newCompIndexSize");
///
///			if (keepval && GetSizeAllocated() > 0)
///			{
///				assert(GetCompPosSize() > 0 && GetSizeAllocated() < numNzElem);
///				memcpy(blockVal, Data(), NzSize());
///				memcpy(blockIds, GetBlockIds(), sizeof(size_t) * GetCompPosSize());
///			}
///			delete[] Buffer();
///			delete[] GetBlockIds();
///
///			SetBuffer(blockVal, numNzElem, false);
///			SetBlockIds(blockIds);
///		}
///		SetSizeAllocated(numNzElem);
///		SetCompPosSize(newCompIndexSize);
///	}
///}
///
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::RequireSizeAndAllocate(size_t rows, size_t cols, size_t nz, bool grow, bool keepval)
///{
///	RequireSizeAndAllocate(numRows, numCols, numNzElem, GetFormat(), growOnly, keepval);
///
///	Resize(rows, cols);
///	if (nz>0) Allocate(nz);
///}
///
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::RequireSizeAndAllocate(size_t numRows, size_t numCols, size_t numNzElem, MatrixFormat mft, bool growOnly, bool keepval)
///{
///	RequireSize(numRows, numCols, numNzElem, mft, growOnly);
///
///	size_t newCompIndexSize = (numCols > numRows ? numCols : numRows) + 1;
///	bool reallocate = (GetSizeAllocated() < numNzElem || 
///					(GetSizeAllocated() > numNzElem && !growOnly) || 
///					GetCompPosSize() < newCompIndexSize);
///	if (reallocate)
///		Allocate(numRows, numCols, numNzElem, growOnly, keepval);
///}
///
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::RequireSize(size_t numRows, size_t numCols, bool growOnly)
///{
///	RequireSize(numRows, numCols, GetFormat(), growOnly);
///}
///
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::RequireSize(size_t numRows, size_t numCols, size_t numNzElem, MatrixFormat mft, bool growOnly)
///{
///	if (GetFormat() != mft || m_numRows != numRows || m_numCols != numCols)
///		Resize(numRows, numCols, numNzElem, mft, growOnly);
///}
///
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::Resize(size_t numRows, size_t numCols, size_t numNz, bool growOnly)
///{
///	Resize(numRows, numCols);
///	if (numNz) Allocate(numNz);
///}
///
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::Resize(size_t numRows, size_t numCols, size_t numNzElem, MatrixFormat mft, bool growOnly)
///{
///	VerifyResizable(__func__);
///
///	m_sliceOffset = 0;
///	m_numRows = numRows;
///	m_numCols = numCols;
///	SetNumStorageRows(numRows);
///	SetNumStorageCols(numCols);
///	SetFormat(mft);
///
///	size_t newCompIndexSize = (numCols > numRows ? numCols : numRows) + 1;
///	bool reallocate = (GetCompPosSize() < newCompIndexSize);
///
///	if (reallocate) Allocate(numRows, numCols, numNzElem, growOnly, false);
///	else Reset();
///}
///
///
//// Reset matrix to 0.
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::Reset()
///{
///	// This is equivalent to setting m_nz = 0; Note we can only do this for sparse CSC/CSR because CompIndexSize is overloaded.
///	if (GetFormat() == matrixFormatSparseCSC || GetFormat() == matrixFormatSparseCSR)
///		memset(GetCompPos(), 0, sizeof(index_t) * GetCompPosSize());
///	SetColIdx(-1);
///	SetBlockSize(0);
///	SetBlockIdShift(0);
///}
///
//// Implements product of one sparse and one dense matrix updating a third dense matrix. Input matrices are optionally transposed.
//// NOTE: The only for using a class template instead of a function template was that I couldn't make the function template compile.
///template <class ElemType, bool denseTimesSparse /* false means SparseTimesDense */, bool transposeA, bool transposeB>
///class MultiplyDenseAndSparse{
///public:
///	// Note: Below the ordering of the matrix parameters 'sparse' and 'dense' does not imply the order of the matrices in the product which is instead controlled
///	// by the value of the boolean template parameter 'denseTimesSparse'.
///	static void MultiplyAndWeightedAdd(ElemType alpha, const CPUSparseMatrix<ElemType>& sparse, const CPUMatrix<ElemType>& dense, ElemType beta, CPUMatrix<ElemType>& c)
///	{
///		const BaseMatrix<ElemType>* lhs = denseTimesSparse ? (const BaseMatrix<ElemType>*) &dense  : (const BaseMatrix<ElemType>*) &sparse;
///		const BaseMatrix<ElemType>* rhs = denseTimesSparse ? (const BaseMatrix<ElemType>*) &sparse : (const BaseMatrix<ElemType>*) &dense;
///
///		// C(m:n) is the product of matrices X * Y where we have the shapes X(m:k) and Y(l:n)
///		size_t m = transposeA ? lhs->GetNumCols() : lhs->GetNumRows();
///		size_t k = transposeA ? lhs->GetNumRows() : lhs->GetNumCols();
///		size_t l = transposeB ? rhs->GetNumCols() : rhs->GetNumRows();
///		size_t n = transposeB ? rhs->GetNumRows() : rhs->GetNumCols();
///
///		if (k != l)
///			InvalidArgument("CPUSparseMatrix::MultiplyAndWeightedAdd: The inner dimensions of a (= %lu) and b (= %lu) don't match.", k, l);
///
///		// Determine the dimension of the outer index of the dense matrix.
///		size_t outerDimensionDense;
///		if      ( denseTimesSparse && !transposeA) outerDimensionDense = dense.GetNumRows();
///		else if ( denseTimesSparse &&  transposeA) outerDimensionDense = dense.GetNumCols();
///		else if (!denseTimesSparse && !transposeB) outerDimensionDense = dense.GetNumCols();
///		else if (!denseTimesSparse &&  transposeB) outerDimensionDense = dense.GetNumRows();
///
///		if (beta == 0) c.Resize(m, n);
///		else c.VerifySize(m, n); // Can't resize if beta != 0
///
///		if (beta == 0)
///			memset(c.Data(), 0, sizeof(ElemType)* c.GetNumElements());
///		else if (beta != 1)
///		{
///#pragma omp parallel for
///			foreach_coord(i, j, c)
///			{
///				c(i, j) = beta * c(i, j);
///			}
///		}
///		else /* beta == 1*/
///			; // We keep the previous value of c before adding the matrix product.
///
///		// In case one factor in the matrix product is empty there is nothing to add to the output c so we can exit here.
///		if (sparse.IsEmpty() || dense.IsEmpty()) return;
///
///		// TODO: Implement CSR as a transposition of b, like we do for GPU.
///		if (sparse.GetFormat() != matrixFormatSparseCSC)
///			NOT_IMPLEMENTED;
///
///		// Up to here we have:
///		// * checked that the matrices are compatible in size
///		// * Initialized the output matrix c
///
///		// Now do the actual multiplication.
///		ElemType* valueBuffer = sparse.Buffer() + *sparse.SecondIndexLocation(); // Points to the value buffer of the current view (i.e. buffer containing values of non-zero elements).
///		int* rowIndexBuffer = sparse.MajorIndexLocation();                          // Points to the index buffer of the current view (i.e. buffer containing indices of non-zero elements).
///		size_t iNonzero = 0;                                                           // Number of nonzero elements handled so far for current slice view.
///		int numPreviosNonzero = sparse.SecondIndexLocation()[0];                 // Total number of nonzero values handled in previous slices.
///
///		// Loop over columns of the sparse matrix
///		for (size_t colSparse = 0; colSparse < sparse.GetNumCols(); colSparse++)
///		{
///			size_t numNonzeroInSparseCol = sparse.SecondIndexLocation()[colSparse + 1] - numPreviosNonzero;
///			// Loop over the nonzero rows of the current column of the sparse matrix
///			for (; iNonzero < numNonzeroInSparseCol; iNonzero++)
///			{
///				size_t rowSparse = rowIndexBuffer[iNonzero]; // RowLocation
///				ElemType sparseVal = valueBuffer[iNonzero];
///
///				// Determine the index of the 'outer' dimension of the sparse matrix and the common inner index.
///				size_t outerIndexSparse;
///				size_t innerIndex;
///				// Below if-statements are evaluated at compile time.
///				if      ( denseTimesSparse && !transposeB) { outerIndexSparse = colSparse; innerIndex = rowSparse; }
///				else if ( denseTimesSparse &&  transposeB) { outerIndexSparse = rowSparse; innerIndex = colSparse; }
///				else if (!denseTimesSparse && !transposeA) { outerIndexSparse = rowSparse; innerIndex = colSparse; }
///				else if (!denseTimesSparse &&  transposeA) { outerIndexSparse = colSparse; innerIndex = rowSparse; }
///
///				// Loop over the outer index of the dense matrix
///				for (size_t outerIndexDense = 0; outerIndexDense < outerDimensionDense; outerIndexDense++)
///				{
///					// Determine the row index of the dense input matrix.
///					// Below if-statements are evaluated at compile time.
///					ElemType denseVal;
///					if      ( denseTimesSparse && !transposeA) denseVal = dense(outerIndexDense,      innerIndex);
///					else if ( denseTimesSparse &&  transposeA) denseVal = dense(     innerIndex, outerIndexDense);
///					else if (!denseTimesSparse && !transposeB) denseVal = dense(     innerIndex, outerIndexDense);
///					else if (!denseTimesSparse &&  transposeB) denseVal = dense(outerIndexDense,      innerIndex);
///
///
///					// Update matrix c.
///					if (denseTimesSparse)
///						c(outerIndexDense, outerIndexSparse) += alpha * denseVal * sparseVal;
///					else /*Sparse times dense */
///						c(outerIndexSparse, outerIndexDense) += alpha * denseVal * sparseVal;
///				}
///			}
///		}
///	}
///};
///
//// c = alpha * lhs * rhs + beta * c
//// dense * sparse -> dense
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::MultiplyAndWeightedAdd(ElemType alpha, const CPUMatrix<ElemType>& a, bool transposeA,
///													const CPUSparseMatrix<ElemType>& b, bool transposeB, ElemType beta, CPUMatrix<ElemType>& c)
///{
///	// Mapping variables to compile time template parameters for efficiency
///	if      ( transposeA &&  transposeB)
///		MultiplyDenseAndSparse<ElemType, true /* dense times sparse */,  true /* transposeA */,  true  /*transposeB*/>::MultiplyAndWeightedAdd(alpha, b /*sparse*/, a /* dense */, beta, c /* matrix beeing updated */);
///	else if ( transposeA && !transposeB)
///		MultiplyDenseAndSparse<ElemType, true /* dense times sparse */,  true /* transposeA */, false  /*transposeB*/>::MultiplyAndWeightedAdd(alpha, b /*sparse*/, a /* dense */, beta, c /* matrix beeing updated */);
///	else if (!transposeA &&  transposeB)
///		MultiplyDenseAndSparse<ElemType, true /* dense times sparse */, false /* transposeA */,  true  /*transposeB*/>::MultiplyAndWeightedAdd(alpha, b /*sparse*/, a /* dense */, beta, c /* matrix beeing updated */);
///	else if (!transposeA && !transposeB)
///		MultiplyDenseAndSparse<ElemType, true /* dense times sparse */, false /* transposeA */, false  /*transposeB*/>::MultiplyAndWeightedAdd(alpha, b /*sparse*/, a /* dense */, beta, c /* matrix beeing updated */);
///}
///
//// c = alpha * lhs * rhs + beta * c
//// sparse * dense -> dense
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::MultiplyAndWeightedAdd(ElemType alpha, const CPUSparseMatrix<ElemType>& a, bool transposeA,
///	const CPUMatrix<ElemType>& b, bool transposeB, ElemType beta, CPUMatrix<ElemType>& c)
///{
///	// Mapping variables to compile time template parameters for efficiency
///	if (transposeA &&  transposeB)
///		MultiplyDenseAndSparse<ElemType, false /* dense times sparse */,  true /* transposeA */,  true /*transposeB*/>::MultiplyAndWeightedAdd(alpha, a /*sparse*/, b /* dense */, beta, c /* matrix beeing updated */);
///	else if (transposeA && !transposeB)
///		MultiplyDenseAndSparse<ElemType, false /* dense times sparse */,  true /* transposeA */, false /*transposeB*/>::MultiplyAndWeightedAdd(alpha, a /*sparse*/, b /* dense */, beta, c /* matrix beeing updated */);
///	else if (!transposeA &&  transposeB)
///		MultiplyDenseAndSparse<ElemType, false /* dense times sparse */, false /* transposeA */,  true /*transposeB*/>::MultiplyAndWeightedAdd(alpha, a /*sparse*/, b /* dense */, beta, c /* matrix beeing updated */);
///	else if (!transposeA && !transposeB)
///		MultiplyDenseAndSparse<ElemType, false /* dense times sparse */, false /* transposeA */, false /*transposeB*/>::MultiplyAndWeightedAdd(alpha, a /*sparse*/, b /* dense */, beta, c /* matrix beeing updated */);
///}
///
//// c += alpha * lhs * rhs
//// sparse += alpha * dense * sparse
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::MultiplyAndAdd(ElemType alpha, const CPUMatrix<ElemType>& a, bool transposeA,
///																const CPUSparseMatrix<ElemType>& b, bool transposeB,
///																CPUSparseMatrix<ElemType>& c)
///{
///	if (a.IsEmpty() || b.IsEmpty())
///		LogicError("CPUSparseMatrix::MultiplyAndAdd; A or B matrix is empty");
///	if (b.GetFormat() != matrixFormatSparseCSC)
///		LogicError("CPUSparseMatrix::MultiplyAndAdd; B matrix is not SparseCSC");
///
///	size_t anr = (transposeA) ? a.GetNumCols() : a.GetNumRows();
///	size_t anc = (transposeA) ? a.GetNumRows() : a.GetNumCols();
///	size_t bnr = (transposeB) ? b.GetNumCols() : b.GetNumRows();
///	size_t bnc = (transposeB) ? b.GetNumRows() : b.GetNumCols();
///
///	// A(k,m) * B(m,n) --> C(k,n)
///	if (anc != bnr)
///		InvalidArgument("CPUSparseMatrix::MultiplyAndAdd; The inner dimensions of a (%lu) and b (%lu) don't match.", anc, bnr);
///
///	if (c.HasExternalBuffer())
///		LogicError("CPUSparseMatrix::MultiplyAndAdd; Cannot modify external buffer in matrix");
///
///	if (c.IsEmpty()) c.Resize(anr,bnc);
///	else if (anr!=c.GetNumRows() || bnc!=c.GetNumCols())
///		InvalidArgument("CPUSparseMatrix::MultiplyAndAdd; Different dimensions of a(,%lu) or b(%lu,) with c(%lu,%lu)", anc, bnr, c.GetNumRows(), c.GetNumCols());
///	c.ConvertToFullBlock();
///
///	ElemType* pdc = c.Data();
///	const ElemType* pda = a.Buffer();
///	const ElemType* pdb = b.Buffer();
///	const index_t* compPos = b.GetPrimePos();
///	const index_t* compId = b.GetCompId();
///	const size_t* blockPos = b.GetBlockPos();
///
///	anr = a.GetNumRows(); anc = a.GetNumCols();
///	bnr = b.GetNumRows(); bnc = b.GetNumCols();
///
///	int m = (transposeA ? 2:0) + (transposeB ? 1:0);
///	if (m==0)
///	{
///		// C += a * A * B
///		for (size_t j=0; j<bnc; ++j)
///		{
///			size_t ns = compPos[j], ne = compPos[j+1]; if (ns==ne) continue;
///			ElemType* pc = pdc + j*anr;
///			for (size_t n=ns; n<ne; ++n)
///			{
///				const ElemType* pa = pda + compId[n]*anr;
///				for (size_t k=0; k<anr; ++k) pc[k] += alpha*pa[k]*pdb[n];
///			}
///		}
///	}
///	else if (m==1)
///	{
///		// C += a * A * Bt
///		for (size_t j=0; j<bnc; ++j)
///		{
///			size_t ns = compPos[j], ne = compPos[j+1]; if (ns==ne) continue;
///			const ElemType* pa = pda + j*anr;
///			for (size_t n=ns; n<ne; ++n)
///			{
///				ElemType* pc = pdc + compId[n]*anr;
///				for (size_t k=0; k<anr; ++k) pc[k] += alpha*pa[k]*pdb[n];
///			}
///		}
///	}
///	else if (m==2)
///	{
///		// C += a * At * B
///		for (size_t j=0; j<bnc; ++j)
///		{
///			size_t ns = compPos[j], ne = compPos[j+1]; if (ns==ne) continue;
///			for (size_t n=ns; n<ne; ++n)
///			{
///				ElemType* pc = pdc + j*anc;
///				const ElemType* pa = pda + compId[n];
///				for (size_t k=0; k<anc; ++k) { pc[k] += alpha*(*pa)*pdb[n]; pa += anr; }
///			}
///		}
///	}
///	else
///	{
///		// C += a * At * Bt
///		for (size_t j=0; j<bnc; ++j)
///		{
///			size_t ns = compPos[j], ne = compPos[j+1]; if (ns==ne) continue;
///			for (size_t n=ns; n<ne; ++n)
///			{
///				ElemType* pc = pdc + compId[n]*anc;
///				const ElemType* pa = pda + j;
///				for (size_t k=0; k<anc; ++k) { pc[k] += alpha*(*pa)*pdb[n]; pa += anr; }
///			}
///		}
///	}
///}
///
////	c[:,j] = alpha * v[j] * a[:,j] + beta * c[:,j]
////	dense = alpha * vector * sparse + beta * dense
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::ColumnwiseScaleAndWeightedAdd(ElemType alpha, const CPUSparseMatrix<ElemType>& a, const CPUMatrix<ElemType>& v, ElemType beta, CPUMatrix<ElemType>& c)
///{
///	if (v.GetNumRows() != 1 && v.GetNumCols() != 1) InvalidArgument("the argument v must be a vector");
///	if (v.GetNumElements() != a.GetNumCols()) InvalidArgument("different size a[*,%d] and v[%d]", int(a.GetNumCols()), int(v.GetNumElements()));
///	//if (a.GetFormat() != matrixFormatSparseCSC) NOT_IMPLEMENTED;
///
///	if (beta == 0) c.Reset();
///	else { c.VerifySize(a.GetNumRows(), a.GetNumCols()); c *= beta; }
///	if (alpha == 0) return;
///
///	MatrixFormat mft = a.GetFormat();
///	int rmf = mft & matrixFormatRowMajor;
///	size_t nc = (rmf) ? a.GetNumRows() : a.GetNumCols();
///	size_t nr = (rmf) ? a.GetNumCols() : a.GetNumRows();
///
///	const ElemType* pv = v.GetData();
///	const ElemType* src = a.Buffer();
///	ElemType* dst = c.Data();
///
///	if (mft & matrixFormatBlock)
///	{
///		// block
///		const size_t* blockPos = a.GetBlockPos();
///		for (size_t j=0; j<nc; ++j)
///		{
///			size_t k = blockPos[j]; if (k==string::npos) continue;
///			const ElemType* pa = src + k*nr;
///			if (rmf)
///			{
///				ElemType* pc = dst + j;
///				for (size_t i=0; i<nr; ++i) { *pc += alpha*pv[i]*(*pa++); pc += nc; }
///			}
///			else
///			{
///				ElemType* pc = dst + j*nr;
///				for (size_t i=0; i<nr; ++i) *pc++ += alpha*pv[j]*(*pa++);
///			}
///		}
///	}
///	else
///	{
///		// sparse
///		const index_t* compPos = a.GetCompPos();
///		const index_t* compId = a.GetCompId();
///		for (size_t j=0; j<nc; ++j)
///		{
///			size_t ns = compPos[j], ne = compPos[j+1];
///			if (rmf)
///			{
///				ElemType* pc = dst + j;
///				for (size_t k=ns; k<ne; ++k) { size_t i = compId[k]; pc[i*nc] += alpha*pv[i]*src[k]; }
///			}
///			else
///			{
///				ElemType* pc = dst + j*nr;
///				for (size_t k=ns; k<ne; ++k) { size_t i = compId[k]; pc[i] += alpha*pv[j]*src[k]; }
///			}
///		}
///	}
///
///	const ElemType* vd = v.Data();
///
///#pragma omp parallel for
///	for (long col = 0; col < (long)a.GetNumCols(); col++)
///	{
///		auto start = a.SecondIndexLocation()[col];
///		auto end = a.SecondIndexLocation()[col + 1];
///
///		for (auto p = start; p < end; p++)
///		{
///			auto row = a.MajorIndexLocation()[p];
///			ElemType val = a.Buffer()[p];
///
///			if (beta == 0) // don't even read the memory if beta is 0
///				c(row, col) = alpha * vd[col] * val;
///			else
///				c(row, col) = alpha * vd[col] * val + beta * c(row, col);
///		}
///	}
///}
///
//// sparse *= alpha
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::Scale(ElemType alpha, CPUSparseMatrix<ElemType>& rhs)
///{
///	if (rhs.IsEmpty())
///		LogicError("Scale: the input sparse matrix is empty");
///
///	MatrixFormat mft = rhs.GetFormat();
///	if (mft == matrixFormatSparseCSC || mft == matrixFormatSparseCSR)
///	{
///		index_t* primePos = rhs.GetPrimePos();
///		size_t nc = (mft & matrixFormatRowMajor) ? rhs.GetNumRows() : rhs.GetNumCols();
///		size_t ns = primePos[0], ne = primePos[nc];
///		ElemType* p = rhs.GetBuffer() + ns;
///		for (size_t j=ns; j<ne; ++j) *p++ *= alpha;
///	}
///	else if (mft == matrixFormatSparseBSC || mft == matrixFormatSparseBSR)
///	{
///		ElemType* pBuffer = rhs.GetBuffer();
///		size_t* blockPos = rhs.GetBlockPos();
///		size_t nc = (mft & matrixFormatRowMajor) ? rhs.GetNumRows() : rhs.GetNumCols();
///		size_t nr = (mft & matrixFormatRowMajor) ? rhs.GetNumCols() : rhs.GetNumRows();
///		for (size_t j=0; j<nc; ++j)
///		{
///			if (blockPos[j]==string::npos) continue;
///			ElemType* p = pBuffer + blockPos[j]*nr;
///			for (size_t i=0; i<nr; ++i) *p++ *= alpha;
///		}
///	}
///}
///
//// dense += alpha * sparse
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::ScaleAndAdd(ElemType alpha, const CPUSparseMatrix<ElemType>& lhs, CPUMatrix<ElemType>& rhs)
///{
///	if (lhs.IsEmpty() || rhs.IsEmpty())
///		LogicError("ScaleAndAdd:  one of the input matrix is empty");
///
///	size_t rows = lhs.GetNumRows(), cols = lhs.GetNumCols();
///	if (rows != rhs.GetNumRows() || cols != rhs.GetNumCols())
///		InvalidArgument("CPUSparseMatrix::ScaleAndAdd: The dimensions of a and b must match");
///
///	MatrixFormat mft = lhs.GetFormat();
///	if (mft == matrixFormatSparseCSC || mft == matrixFormatSparseCSR)
///	{
///		index_t* primePos = lhs.GetPrimePos();
///		index_t* compId = lhs.GetCompId();
///		const ElemType* lp = lhs.Buffer();
///
///		bool csc = (mft == matrixFormatSparseCSC);
///		size_t nc = csc ? cols : rows;
///		for (size_t j=0; j<nc; ++j)
///		{
///			size_t ns = primePos[j], ne = primePos[j+1];
///			for (size_t n=ns; n<ne; ++n)
///			{
///				size_t i = compId[n];
///				if (csc) rhs(i,j) += alpha * lp[n];
///				else rhs(j,i) += alpha * lp[n];
///			}
///		}
///	}
///	else if (mft == matrixFormatSparseBSC || mft == matrixFormatSparseBSR)
///	{
///		size_t* blockPos = lhs.GetBlockPos();
///		const ElemType* lp = lhs.Buffer();
///
///		bool bsc = (mft == matrixFormatSparseBSC);
///		size_t nc = bsc ? cols : rows;
///		size_t nr = bsc ? rows : cols;
///		for (size_t j=0; j<nc; ++j)
///		{
///			if (blockPos[j]==string::npos) continue;
///			const ElemType* p = lp + blockPos[j]*nr;
///			if (bsc) for (size_t i=0; i<nr; ++i) rhs(i,j) += alpha * (*p++);
///			else for (size_t i=0; i<nr; ++i) rhs(j,i) += alpha * (*p++);
///		}
///	}
///	else
///	{
///		RuntimeError("CPUSparseMatrix:: ScaleAndAdd() Not implemented");
///	}
///}
///
///template <class ElemType>
///bool CPUSparseMatrix<ElemType>::AreEqual(const CPUSparseMatrix<ElemType>& a, const CPUSparseMatrix<ElemType>& b, ElemType threshold)
///{
///	if (a.IsEmpty() || b.IsEmpty())
///		LogicError("AreEqual: one of the input matrices is empty");
///
///	if (a.GetNumRows() != b.GetNumRows() || a.GetNumCols() != b.GetNumCols())
///		return false;
///
///	bool result = true;
///
///#pragma omp parallel for
///	foreach_coord (i, j, a)
///	{
///		if (abs(a(i, j) - b(i, j)) > threshold) { result = false; break; }
///	}
///	return result;
///}
///
///template<class ElemType>
///void CPUSparseMatrix<ElemType>::InnerProduct(const CPUSparseMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c, bool isColWise)
///{
///	if (a.IsEmpty() || b.IsEmpty())
///		LogicError("InnerProduct:  one of the input matrices is empty");
///
///	const int m = (int)a.GetNumRows();
///	const int n = (int)a.GetNumCols();
///	const int k = (int)b.GetNumRows();
///	const int l = (int)b.GetNumCols();
///
///	assert(m > 0 && n > 0 && k > 0 && l > 0); // converting from size_t to int may cause overflow
///	assert(m == k && n == l);                 // converting from size_t to int may cause overflow
///	if (m != k || n != l)
///		InvalidArgument("InnerProduct: Matrices a and b should have same dimension");
///
///	if (isColWise) // col-wise
///	{
///		c.Resize(1, n);
///
///#pragma omp parallel for
///		foreach_column(j, c)
///		{
///			ElemType sum = 0;
///			for (index_t iRow = a.ColLocation()[j]; iRow < a.ColLocation()[j+1]; ++iRow)
///			{
///				size_t row = a.RowLocation()[iRow];
///				sum += a.Data()[iRow] * b(row, j);
///			}
///			c(0, j) = sum;
///		}
///	}
///	else
///	{
///		c.Resize(m, 1);
///
///#pragma omp parallel for
///		foreach_row(i, c)
///		{
///			ElemType sum = 0;
///			for(index_t j = 0; j < n; ++j)
///			{
///				for (index_t iRow = a.ColLocation()[j]; iRow < a.ColLocation()[j + 1]; ++iRow)
///				{
///					if (a.RowLocation()[iRow] == i)
///					{
///						sum += a.Data()[iRow] * b(i, j);
///						break;
///					}
///				}
///			}
///			c(i, 0) = sum;
///		}
///	}
///}
///
//// A helper method used in MomentumSGDUpdate and NesterovAcceleratedMomentumSGDUpdate.
//// Modifies the smoothed gradients "c", as well as the current gradients "this" on which this method is invoked.
//// Classic momentum (unitGainFactor == 1.0):
//// 1) c = momentum * c + this
//// Unit-gain momentum (unitGainFactor == 1.0 - momentum):
//// 1) c = momentum * c + (1.0 - momentum) * this
//// 2) this = c
//// TODO: NormalGrad is a misnomer here. Come up with a better name.
///template <class ElemType>
///void CPUSparseMatrix<ElemType>::NormalGrad(CPUMatrix<ElemType>& c, ElemType momentum, ElemType unitGainFactor)
///{
///	if (c.IsEmpty())
///	{
///		c.Resize(GetNumRows(), GetNumCols());
///		c.SetValue(0.0);
///	}
///	// BUGBUG: dimension/ownbuffer check?
///
///	if (GetFormat() == matrixFormatSparseBSC || GetFormat() == matrixFormatSparseBSR)
///	{
///		const auto isSparseBlockCol = (GetFormat() == matrixFormatSparseBSC);
///		for (size_t j = 0; j < GetBlockSize(); j++)
///		{
///			size_t i = GetBlockIds()[j] - GetBlockIdShift();
///			size_t len = (isSparseBlockCol) ? GetNumRows() : GetNumCols();
///			size_t start = j*len;
///			for (size_t p = start; p < start + len; p++)
///			{
///				ElemType val = Buffer()[p];
///				size_t row = (isSparseBlockCol) ? (p - start) : i;
///				size_t col = (isSparseBlockCol) ? i : (p - start);
///				c(row, col) = unitGainFactor * val + momentum * c(row, col);
///				Buffer()[p] = c(row, col);
///			}
///		}
///	}
///	else
///	{
///		RuntimeError("CPUSparseMatrix:: NormalGrad() only support block sparse format");
///	}
///}
///
//// update smoothed gradients c and current gradients (this)
///template <class ElemType>
///ElemType CPUSparseMatrix<ElemType>::Adagrad(CPUMatrix<ElemType>& c, bool needAveMultiplier)
///{
///	if (c.IsEmpty() || c.GetNumCols() != GetNumCols() || c.GetNumRows() != GetNumRows())
///	{
///		c.Resize(GetNumRows(), GetNumCols());
///		c.SetValue(0.0);
///	}
///	// BUGBUG: dimension/ownbuffer check?
///
///	ElemType aveMultiplier = 0;
///
///	ElemType floor = 1e-16f;
///	if (GetFormat() == matrixFormatSparseCSC || GetFormat() == matrixFormatSparseCSR)
///	{
///		size_t col_num = (GetFormat() == matrixFormatSparseCSC) ? GetNumCols() : GetNumRows();
///		for (size_t j = 0; j < col_num; j++)
///		{
///			size_t start = SecondIndexLocation()[j];
///			size_t end = SecondIndexLocation()[j + 1];
///			for (size_t p = start; p < end; p++)
///			{
///				size_t i = MajorIndexLocation()[p];
///				ElemType val = Buffer()[p];
///
///				size_t row = (GetFormat() == matrixFormatSparseCSC) ? i : j;
///				size_t col = (GetFormat() == matrixFormatSparseCSC) ? j : i;
///				ElemType adenorm = c(row, col);
///				adenorm += val * val;
///				ElemType a = sqrt(floor + adenorm);
///				Buffer()[p] = val / a;
///				c(row, col) = adenorm;
///
///				if (needAveMultiplier)
///					aveMultiplier += 1 / a;
///			}
///		}
///	}
///	else if (GetFormat() == matrixFormatSparseBSC || GetFormat() == matrixFormatSparseBSR)
///	{
///		size_t len = (GetFormat() == matrixFormatSparseBSC) ? GetNumRows() : GetNumCols();
///		size_t p = 0;
///		for (long j = 0; j < GetBlockSize(); j++)
///		{
///			size_t colOrRow = GetBlockIds()[j] - GetBlockIdShift();
///			for (long i = 0; i < len; i++, p++)
///			{
///				ElemType val = Buffer()[p];
///
///				size_t row = (GetFormat() == matrixFormatSparseBSC) ? i : colOrRow;
///				size_t col = (GetFormat() == matrixFormatSparseBSC) ? colOrRow : i;
///				c(row, col) += val * val;
///				ElemType a = sqrt(floor + c(row, col));
///				Buffer()[p] /= a;
///				if (needAveMultiplier) aveMultiplier += 1 / a;
///			}
///		}
///	}
///	size_t nz = NzCount();
///	if (needAveMultiplier && nz > 0) return aveMultiplier / nz;
///	return 1;
///}
///
///template <class ElemType>
///template <class AccumType>
///void CPUSparseMatrix<ElemType>::AdaDelta(CPUMatrix<AccumType>& c, CPUMatrix<AccumType>& functionValues, AccumType learningRate, AccumType rho, AccumType epsilon, int* timestamps, int currentTimestamp)
///{
///	size_t numColsNeeded = 2 * GetNumCols();
///
///	if (c.IsEmpty() || (c.GetNumCols() < numColsNeeded))
///	{
///		c.Resize(GetNumRows(), numColsNeeded);
///		c.SetValue(0.0);
///	}
///
///	if (c.GetNumRows() != GetNumRows() || c.GetNumCols() != numColsNeeded)
///		LogicError("The matrix gradients does not have expected dimensions");
///
///	if (GetFormat() != matrixFormatSparseBSC)
///		LogicError("Unsupported sparse format");
///
///	size_t n = GetNumElements();
///	ElemType* grad = Data();
///	AccumType* smoothAda = c.Data();
///	AccumType* smoothX2 = c.Data() + n;
///	AccumType* val = functionValues.Data();
///	auto rows = GetNumRows();
///
///#pragma omp parallel for
///	// TODO: Unroll 4-times for better performance leveraging vectorization
///	for (auto blockid = 0; blockid < (int)GetBlockSize(); ++blockid)
///	{
///		auto col = GetBlockIds()[blockid] - GetBlockIdShift();
///		auto columnOffset = col * rows;
///		auto blockOffset = blockid * rows;
///		auto decay = std::pow(rho, currentTimestamp - 1 - timestamps[col]);
///		timestamps[col] = currentTimestamp;
///		for (auto row = 0; row < rows; ++row)
///		{
///			size_t denseIndex = columnOffset + row;;
///			ElemType g = grad[blockOffset + row];
///			AccumType adaSqr = rho * decay * smoothAda[denseIndex] + (1 - rho) * g * g;
///			smoothAda[denseIndex] = adaSqr;
///			AccumType x2 = decay * smoothX2[denseIndex];
///			AccumType deltaX = -sqrt(x2 + epsilon) / sqrt(adaSqr + epsilon) * g;
///			smoothX2[denseIndex] = rho * x2 + (1 - rho) * deltaX * deltaX;
///			val[denseIndex] += learningRate * deltaX;
///		}
///	}
///}
///
///template <class ElemType>
///CPUSparseMatrix<ElemType>& CPUSparseMatrix<ElemType>::InplaceTruncateTop(ElemType threshold)
///{
///	if (!OwnBuffer())
///		LogicError("Cannot modify since the buffer is managed externally");
///
///	long m = (long) NzCount();
///	ElemType* nzValues = NzValues();
///
///#pragma omp parallel for
///	for (long i = 0; i < (m & ~3); i += 4) // four-way unrolling
///	{
///		if (nzValues[i] > threshold) nzValues[i] = threshold;
///		if (nzValues[i+1] > threshold) nzValues[i+1] = threshold;
///		if (nzValues[i+2] > threshold) nzValues[i+2] = threshold;
///		if (nzValues[i+3] > threshold) nzValues[i+3] = threshold;
///	}
///	// handle remaining stuffs
///	for (long i = m & ~3; i < m; i++)
///		if (nzValues[i] > threshold) nzValues[i] = threshold;
///
///	return *this;
///}
///
///template <class ElemType>
///CPUSparseMatrix<ElemType>& CPUSparseMatrix<ElemType>::InplaceTruncateBottom(ElemType threshold)
///{
///	if (!OwnBuffer())
///		LogicError("Cannot modify since the buffer is managed externally");
///
///	long m = (long) NzCount();
///	ElemType* nzValues = NzValues();
///
///#pragma omp parallel for
///	for (long i = 0; i < (m & ~3); i += 4) // four-way unrolling
///	{
///		if (nzValues[i] < threshold)
///			nzValues[i] = threshold;
///
///		if (nzValues[i+1] < threshold)
///			nzValues[i+1] = threshold;
///
///		if (nzValues[i+2] < threshold)
///			nzValues[i+2] = threshold;
///
///		if (nzValues[i+3] < threshold)
///			nzValues[i+3] = threshold;
///	}
///	// handle remaining stuffs
///	for (long i = m & ~3; i < m; i++)
///	{
///		if (nzValues[i] < threshold)
///			nzValues[i] = threshold;
///	}
///
///	return *this;
///}
///
///template <class ElemType>
///CPUSparseMatrix<ElemType>& CPUSparseMatrix<ElemType>::InplaceTruncate(ElemType threshold)
///{
///	if (!OwnBuffer())
///		LogicError("Cannot modify since the buffer is managed externally");
///
///	ElemType locThresholdPos = abs(threshold);
///	ElemType locTHresholdNeg = -locThresholdPos;
///
///	long m = (long) NzCount();
///	ElemType* nzValues = NzValues();
///
///#pragma omp parallel for
///	for (long i = 0; i < (m & ~3); i += 4) // four-way unrolling
///	{
///		if (nzValues[i] > locThresholdPos)
///			nzValues[i] = locThresholdPos;
///		else if (nzValues[i] < locTHresholdNeg)
///			nzValues[i] = locTHresholdNeg;
///
///		if (nzValues[i+1] > locThresholdPos)
///			nzValues[i+1] = locThresholdPos;
///		else if (nzValues[i+1] < locTHresholdNeg)
///			nzValues[i+1] = locTHresholdNeg;
///
///		if (nzValues[i+2] > locThresholdPos)
///			nzValues[i+2] = locThresholdPos;
///		else if (nzValues[i+2] < locTHresholdNeg)
///			nzValues[i+2] = locTHresholdNeg;
///
///		if (nzValues[i+3] > locThresholdPos)
///			nzValues[i+3] = locThresholdPos;
///		else if (nzValues[i+3] < locTHresholdNeg)
///			nzValues[i+3] = locTHresholdNeg;
///	}
///	// handle remaining stuffs
///	for (long i = m & ~3; i < m; i++)
///	{
///		if (nzValues[i] > locThresholdPos)
///			nzValues[i] = locThresholdPos;
///		else if (nzValues[i] < locTHresholdNeg)
///			nzValues[i] = locTHresholdNeg;
///	}
///
///	return *this;
///}
///
///template <class ElemType>
///CPUSparseMatrix<ElemType>& CPUSparseMatrix<ElemType>::InplaceSoftThreshold(ElemType threshold)
///{
///	if (!OwnBuffer())
///		LogicError("Cannot modify since the buffer is managed externally");
///
///	long m = (long) NzCount();
///	ElemType* nzValues = NzValues();
///
///#pragma omp parallel for
///	for (long i = 0; i < (m & ~3); i += 4) // four-way unrolling
///	{
///		if (nzValues[i] > threshold)
///			nzValues[i] -= threshold;
///		else if (nzValues[i] < -threshold)
///			nzValues[i] += threshold;
///		else
///			nzValues[i] = 0;
///
///		if (nzValues[i+1] > threshold)
///			nzValues[i+1] -= threshold;
///		else if (nzValues[i+1] < -threshold)
///			nzValues[i+1] += threshold;
///		else
///			nzValues[i+1] = 0;
///
///		if (nzValues[i+2] > threshold)
///			nzValues[i+2] -= threshold;
///		else if (nzValues[i+2] < -threshold)
///			nzValues[i+2] += threshold;
///		else
///			nzValues[i+2] = 0;
///
///		if (nzValues[i+3] > threshold)
///			nzValues[i+3] -= threshold;
///		else if (nzValues[i+3] < -threshold)
///			nzValues[i+3] += threshold;
///		else
///			nzValues[i+3] = 0;
///	}
///	// handle remaining stuffs
///	for (long i = m & ~3; i < m; i++)
///	{
///		if (nzValues[i] > threshold)
///			nzValues[i] -= threshold;
///		else if (nzValues[i] < -threshold)
///			nzValues[i] += threshold;
///		else
///			nzValues[i] = 0;
///	}
///	return *this;
///}
///
///template <class ElemType>
///ElemType CPUSparseMatrix<ElemType>::FrobeniusNorm() const
///{
///	if (IsEmpty()) return 0;
///
///	ElemType v = 0; // TODO: do this in 'double'?
///
///	long m = (long) NzCount();
///	const ElemType* nzValues = NzValues();
///
////four-way unrolling
///#pragma omp parallel for reduction(+ : v)
///	for (long i = 0; i < (m & ~3); i += 4)
///	{
///		v += nzValues[i] * nzValues[i] + nzValues[i+1] * nzValues[i+1] + nzValues[i+2] * nzValues[i+2] + nzValues[i+3] * nzValues[i+3];
///	}
///	// handle remaining stuffs
///	for (long i = m & ~3; i < m; i++)
///	{
///		v += nzValues[i] * nzValues[i];
///	}
///
///	return sqrt(v);
///}
///
////sum of all abs(elements)
///template <class ElemType>
///ElemType CPUSparseMatrix<ElemType>::SumOfAbsElements() const
///{
///	if (IsEmpty()) return 0;
///
///	if (sizeof(ElemType) == sizeof(double))
///		return (ElemType) cblas_dasum((int) NzCount(), reinterpret_cast<double*>(Data()), 1);
///#pragma warning(suppress : 4244)
///	return cblas_sasum((int) NzCount(), reinterpret_cast<float*>(Data()), 1);
///}
///
////sum of all elements
///template <class ElemType>
///ElemType CPUSparseMatrix<ElemType>::SumOfElements() const
///{
///	if (IsEmpty()) return 0;
///
///	ElemType sum = 0; // TODO: Do this in 'double'?
///
///	long m = (long) NzCount();
///	const ElemType* nzValues = NzValues();
///
////four-way unrolling
///#pragma omp parallel for reduction(+ : sum)
///	for (long i = 0; i < (m & ~3); i += 4)
///	{
///		sum += nzValues[i] + nzValues[i+1] + nzValues[i+2] + nzValues[i+3];
///	}
///	// handle remaining stuffs
///	for (long i = m & ~3; i < m; i++)
///	{
///		sum += nzValues[i];
///	}
///
///	return sum;
///}
///
//// specialization to RunTimeError for now due to omp implementation only support build-in type
///template <>
///half CPUSparseMatrix<half>::FrobeniusNorm() const
///{
///	RuntimeError("half FrobeniusNorm not supported");
///}
///template <>
///half CPUSparseMatrix<half>::SumOfElements() const
///{
///	RuntimeError("half SumOfElements not supported");
///}
///
///template <typename ElemType>
///MATH_API File& operator>>(File& stream, CPUSparseMatrix<ElemType>& us)
///{
///	if (!us.OwnBuffer())
///		LogicError("Cannot read into a managed external matrix");
///
///	stream.GetMarker("BMAT");
///	size_t elsize; stream >> elsize;
///	if (sizeof(ElemType) != elsize)
///		RuntimeError("Template argument size doesn't match those in file");
///
///	int format;
///	std::string matrixName;
///	size_t nz, colnum, rownum;
///	stream >> matrixName >> format >> nz >> colnum >> rownum;
///
///	us.SetFormat((MatrixFormat) format);
///	if (us.GetFormat() != matrixFormatSparseCSC && us.GetFormat() != matrixFormatSparseCSR)
///		NOT_IMPLEMENTED;
///
///	us.RequireSizeAndAllocate(rownum, colnum, nz, true, false);
///
///	if (nz > 0)
///	{
///		size_t compressedSize = (us.GetFormat() == matrixFormatSparseCSC) ? colnum + 1 : rownum + 1;
///		ElemType* dataBuffer = us.NzValues();
///		index_t* unCompressedIndex = us.MajorIndexLocation();
///		index_t* compressedIndex = us.SecondIndexLocation();
///
///		// read in the sparse matrix info
///		for (size_t i = 0; i < nz; ++i) stream >> dataBuffer[i];
///		for (size_t i = 0; i < nz; ++i) stream >> unCompressedIndex[i];
///		for (size_t i = 0; i < compressedSize; ++i) stream >> compressedIndex[i];
///	}
///	stream.GetMarker("EMAT");
///
///	return stream;
///}
///
///template MATH_API File& operator>>(File& stream, CPUSparseMatrix<float>& us);
///template MATH_API File& operator>>(File& stream, CPUSparseMatrix<double>& us);
///
///template <typename ElemType>
///MATH_API File& operator<<(File& stream, const CPUSparseMatrix<ElemType>& us)
///{
///	if (us.GetFormat() != matrixFormatSparseCSC && us.GetFormat() != matrixFormatSparseCSR)
///		NOT_IMPLEMENTED;
///
///	stream.PutMarkerBegin("BMAT");
///	stream << sizeof(ElemType);
///	stream << "nnmatrix";		// needed for compatability, and could potentially be an empty string
///
///	int format = us.GetFormat();
///	size_t nz, numRows, numCols;
///	size_t compressedSize = us.SecondIndexCount();
///	stream << format << nz << numCols << numRows;
///
///	if (nz > 0)
///	{
///		ElemType* dataBuffer = us.NzValues();
///		index_t* unCompressedIndex = us.MajorIndexLocation();
///		index_t* compressedIndex = us.SecondIndexLocation();
///
///		for (size_t i = 0; i < nz; ++i) stream << dataBuffer[i];
///		for (size_t i = 0; i < nz; ++i) stream << unCompressedIndex[i];
///		for (size_t i = 0; i < compressedSize; ++i) stream << compressedIndex[i];
///	}
///	stream.PutMarkerEnd("EMAT");
///
///	return stream;
///}

///template class CPUSparseMatrix<half>;
///template class CPUSparseMatrix<float>;
///template class CPUSparseMatrix<double>;


// instantiate learner methods
//template void CPUSparseMatrix<float>::AdaDelta(CPUMatrix<float>& c, CPUMatrix<float>& functionValues, float learningRate, float rho, float epsilon, int* timestamps, int currentTimestamp);
//template void CPUSparseMatrix<double>::AdaDelta(CPUMatrix<double>& c, CPUMatrix<double>& functionValues, double learningRate, double rho, double epsilon, int* timestamps, int currentTimestamp);
//template void CPUSparseMatrix<half>::AdaDelta(CPUMatrix<float>& c, CPUMatrix<float>& functionValues, float learningRate, float rho, float epsilon, int* timestamps, int currentTimestamp);

//	<char>
///template CPUSparseMatrix<char>::CPUSparseMatrix(MatrixFormat mft, size_t numRows, size_t numCols, size_t size);
//template CPUSparseMatrix<char>::CPUSparseMatrix(MatrixFormat);
//template CPUSparseMatrix<char>::CPUSparseMatrix(CPUSparseMatrix<char> const&);
//template CPUSparseMatrix<char>::CPUSparseMatrix(CPUSparseMatrix<char>&&);
//template CPUSparseMatrix<char>& CPUSparseMatrix<char>::operator=(CPUSparseMatrix<char>&& moveFrom);
//template void CPUSparseMatrix<char>::SetValue(size_t, size_t, char);
//template void CPUSparseMatrix<char>::SetValue(CPUMatrix<char> const&);
//template void CPUSparseMatrix<char>::SetValue(GPUMatrix<char> const&);
///template void CPUSparseMatrix<char>::SetValue(CPUSparseMatrix<char> const&);
//template void CPUSparseMatrix<char>::SetValue(GPUSparseMatrix<char> const&);
///template char* CPUSparseMatrix<char>::Data() const;
//template void CPUSparseMatrix<char>::Reset();
///template void CPUSparseMatrix<char>::Resize(size_t, size_t, size_t, const bool);
///template void CPUSparseMatrix<char>::RequireSizeAndAllocate(size_t, size_t, size_t, const bool, bool);
//template void CPUSparseMatrix<char>::RequireSizeAndAllocate(size_t, size_t, size_t, MatrixFormat, const bool, bool);
//template CPUSparseMatrix<char>::~CPUSparseMatrix();
///template CPUSparseMatrix<char> CPUSparseMatrix<char>::GetColumnSlice(size_t startColumn, size_t numCols) const;
///template CPUMatrix<char> CPUSparseMatrix<char>::CopyColumnSliceToDense(size_t startColumn, size_t numCols) const;
///template void CPUSparseMatrix<char>::AssignColumnSliceToDense(CPUMatrix<char>&, size_t startColumn, size_t numCols) const;
///template CPUSparseMatrix<char>& CPUSparseMatrix<char>::operator=(const CPUSparseMatrix<char>& copyFrom);
///template void CPUSparseMatrix<char>::ScaleAndAdd(char, class Microsoft::MSR::CNTK::CPUSparseMatrix<char> const &, class Microsoft::MSR::CNTK::CPUMatrix<char> &);
///template void CPUSparseMatrix<char>::SetMatrixFromCSCFormat(const index_t* h_CSCCol, const index_t* h_Row, const char* h_Val, size_t nz, size_t numRows, size_t numCols);

//	<short>
///template CPUSparseMatrix<short>::CPUSparseMatrix(MatrixFormat mft, size_t numRows, size_t numCols, size_t size);
//template CPUSparseMatrix<short>::CPUSparseMatrix(MatrixFormat);
//template CPUSparseMatrix<short>::CPUSparseMatrix(CPUSparseMatrix<short> const&);
//template CPUSparseMatrix<short>::CPUSparseMatrix(CPUSparseMatrix<short>&&);
//template CPUSparseMatrix<short>& CPUSparseMatrix<short>::operator=(CPUSparseMatrix<short>&& moveFrom);
//template void CPUSparseMatrix<short>::SetValue(size_t, size_t, short);
//template void CPUSparseMatrix<short>::SetValue(CPUMatrix<short> const&);
//template void CPUSparseMatrix<short>::SetValue(GPUMatrix<short> const&);
///template void CPUSparseMatrix<short>::SetValue(CPUSparseMatrix<short> const&);
//template void CPUSparseMatrix<short>::SetValue(GPUSparseMatrix<short> const&);
///template short* CPUSparseMatrix<short>::Data() const;
//template void CPUSparseMatrix<short>::Reset();
///template void CPUSparseMatrix<short>::Resize(size_t, size_t, size_t, const bool);
///template void CPUSparseMatrix<short>::RequireSizeAndAllocate(size_t, size_t, size_t, const bool, bool);
//template void CPUSparseMatrix<short>::RequireSizeAndAllocate(size_t, size_t, size_t, MatrixFormat, const bool, bool);
//template CPUSparseMatrix<short>::~CPUSparseMatrix();
///template CPUSparseMatrix<short> CPUSparseMatrix<short>::GetColumnSlice(size_t startColumn, size_t numCols) const;
///template CPUMatrix<short> CPUSparseMatrix<short>::CopyColumnSliceToDense(size_t startColumn, size_t numCols) const;
///template void CPUSparseMatrix<short>::AssignColumnSliceToDense(CPUMatrix<short>&, size_t startColumn, size_t numCols) const;
///template CPUSparseMatrix<short>& CPUSparseMatrix<short>::operator=(const CPUSparseMatrix<short>& copyFrom);
///template void CPUSparseMatrix<short>::ScaleAndAdd(short, class Microsoft::MSR::CNTK::CPUSparseMatrix<short> const &, class Microsoft::MSR::CNTK::CPUMatrix<short> &);
///template void CPUSparseMatrix<short>::SetMatrixFromCSCFormat(const index_t* h_CSCCol, const index_t* h_Row, const short* h_Val, size_t nz, size_t numRows, size_t numCols);

//	<int>
///template CPUSparseMatrix<int>::CPUSparseMatrix(MatrixFormat, size_t, size_t, size_t);
//template CPUSparseMatrix<int>::~CPUSparseMatrix();

} } }

