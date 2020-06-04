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
//#include <lapacke.h>
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
void CPUSparseMatrix<ElemType>::SetValue(const CPUSparseMatrix<ElemType>& mat)
{
	if (&mat==this) return;
	if (mat.IsColMajor()) Assign(mat);
	else Assign(mat.TransposeTo(CPUSparseMatrix(),true),true);
}

template <class ElemType>
CPUSparseMatrix<ElemType> CPUSparseMatrix<ElemType>::GetColumnSlice(size_t start, size_t len) const
{
	CPUSparseMatrix<ElemType> slice(*this, true);
	slice.SetColumnSlice(start, len);
	return slice;
}

template <class ElemType>
CPUMatrix<ElemType> CPUSparseMatrix<ElemType>::Diagonal() const
{
	size_t n = GetDiagSize();
	CPUMatrix<ElemType> diag(1,n);
	if (n)
	{
		const index_t* compPos = GetPrimePos();
		const ElemType* pi = GetBuffer();
		ElemType* po = diag.GetData();

		if (IsSparseFormat())
		{
			const index_t* compId = GetCompId();
			for (size_t j=0; j<n; ++j,++po)
			{
				size_t ns = compPos[j], ne = compPos[j+1];
				for (size_t k=ns; k<ne; ++k)
					if (compId[k]==j) { *po = pi[k]; break; }
			}
		}
		else for (size_t j=0; j<n; ++j,++po)
		{
			size_t k = compPos[j];
			if (k!=string::npos) *po = pi[k*m_numRows + j];
		}
	}
	return diag;
}

template <class ElemType>
void CPUSparseMatrix<ElemType>::SetDiagonalValue(ElemType v)
{
	Reset();
	size_t dsize = GetDiagSize(); if (dsize==0) return;

	MatrixFormat mft = GetFormat();
	size_t nc = (mft & matrixFormatRowMajor) ? m_numRows : m_numCols;
	size_t nr = (mft & matrixFormatRowMajor) ? m_numCols : m_numRows;

	if (mft & matrixFormatBlock)
	{
		Allocate(dsize*nr);
		index_t* blockPos = GetPrimePos();
		ElemType* p = GetBuffer(); memset(p, 0, dsize*nr*sizeof(ElemType));
		for (size_t j=0; j<dsize; ++j) { blockPos[j] = (index_t)j; p[j] = v; p += nr; }
		m_sob->SetBlockCount(dsize);
	}
	else
	{
		Allocate(max(nr,nc));
		index_t* compPos = GetPrimePos();
		index_t* compId = GetCompId();
		ElemType* p = GetBuffer();
		for (size_t j=0; j<dsize; ++j) { compPos[j] = compId[j] = (index_t)j; *p++ = v; }
		for (size_t j=dsize; j<=nc; ++j) compPos[j] = (index_t)dsize;
	}
}

template <class ElemType>
void CPUSparseMatrix<ElemType>::SetDiagonalValue(const CPUMatrix<ElemType>& v)
{
	if (v.GetNumRows()!=1 && v.GetNumCols()!=1)
		LogicError("SetDiagonalValue: Input vector (%lu,%lu) is not vector", v.GetNumRows(), v.GetNumCols());

	size_t dsize = v.GetNumElements();
	if (GetDiagSize()!=dsize)
		LogicError("SetDiagonalValue: Different diagonal size=%lu / %lu", GetDiagSize(), dsize);

	if (dsize == 0) return;
	if (dsize == 1) { SetDiagonalValue(*v.GetData()); return; }

	Reset();
	MatrixFormat mft = GetFormat();
	if (mft & matrixFormatBlock)
	{
		size_t nr = (mft & matrixFormatRowMajor) ? m_numCols : m_numRows;
		Allocate(dsize*nr);
		const ElemType* src = v.GetData();
		ElemType* dst = GetData();

		index_t* blockPos = GetPrimePos();
		for (size_t j=0; j<dsize; ++j) { blockPos[j] = (index_t)j; dst[j] = *src++; dst += nr; }
		m_sob->SetBlockCount(dsize);
	}
	else
	{
		size_t nc = (mft & matrixFormatRowMajor) ? m_numRows : m_numCols;
		Allocate(dsize);
		const ElemType* pi = v.GetData();
		ElemType* po = GetData();

		index_t* compPos = GetPrimePos();
		index_t* compId = GetCompId();
		for (size_t j=0; j<dsize; ++j) { compPos[j] = compId[j] = (index_t)j; *po++ = *pi++; }
		for (size_t j=dsize; j<=nc; ++j) compPos[j] = (index_t)dsize;
	}
}

template <class ElemType>
CPUSparseMatrix<ElemType>& CPUSparseMatrix<ElemType>::AssignOneHot(const CPUMatrix<ElemType>& a, vector<size_t>& shape, size_t axis)
{
	if (a.IsEmpty()) LogicError("AssignOneHot: Matrix a is empty");
	if (GetFormat() != matrixFormatSparseCSC) LogicError("AssignOneHot: Matrix format is not supported");
	if (axis >= shape.size()) LogicError("AssignOneHot: axis is not correct");

	int itemSize = 1;
	for (size_t i=0; i<axis; ++i) itemSize *= (int)shape[i];
	int numClass = (int)shape[axis];

	size_t nSize = a.GetNumElements();
	size_t nRows = itemSize * numClass;
	size_t nCols = nSize / itemSize;
	//if (m_numRows==0 || m_numRows!=nRows || m_numCols==0 || m_numCols!=nCols)
	//	LogicError("AssignOneHot: Target matrix size is not correct");

	Reset();
	Resize(nRows, nCols);
	Allocate(nSize);

	index_t* compPos = GetPrimePos();
	index_t* compId = GetCompId();
	const ElemType* src = a.GetData();
	ElemType* dst = GetData();
#pragma omp parallel for
	for (long j=0; j<nSize; ++j)
	{
		int blockId = j / itemSize;	// cid
		int itemId = j % itemSize;	// rid
		// for invalid indices, theorically they should not belong to nz elements.
		// but if we scan the indices to count the valid indices number,
		// it will be difficult for parallel calculation, especially on GPU.
		// here we chose to keep those elements in nz element list, but with value 0 an at row 0
		if (src[j]<0 || src[j]>=numClass) { compId[j] = index_t(itemId); dst[j] = 0; }
		else { compId[j] = int(src[j])*itemSize + itemId; dst[j] = 1; }
		if (itemId == 0) compPos[blockId+1] = index_t(itemSize*(blockId+1));
	}
	compPos[0] = 0;

	return *this;
}

///template <class ElemType>
///void CPUSparseMatrix<ElemType>::MaskColumnsValue(const CPUMatrix<char>& mask, ElemType val, size_t mcols)
///{
///	NOT_IMPLEMENTED;
///}

// this[:,j] = a[:,idx[j]] * alpha + this[:,j] * beta
template <class ElemType>
CPUSparseMatrix<ElemType>& CPUSparseMatrix<ElemType>::DoGatherColumnsOf(ElemType alpha, const CPUSparseMatrix<ElemType>& a, const CPUMatrix<ElemType>& idx, ElemType beta)
{
	if (idx.GetNumRows()!=1)
		InvalidArgument("DoGatherColumnsOf; Map must be a row vector");

	if (beta) VerifySize(a.GetNumRows(), idx.GetNumCols(), "DoGatherColumnsOf");
	else { Reset(); Resize(a.GetNumRows(), idx.GetNumCols()); }

	const ElemType* px = idx.GetData();
	size_t nca = a.GetNumCols();

	if (GetItemCount()==0 || beta==0)
	{
		size_t n = a.GetItemCount();
		if (n==0 || alpha==0) return *this;

		Allocate(n);
		const ElemType* pBuffer = a.GetBuffer();
		const index_t* compPos = a.GetPrimePos();
		const index_t* compId = a.GetCompId();
		for (size_t j=0; j<m_numCols; ++j)
		{
			if (std::isnan(*px) || *px < 0) { ++px; continue; }
			size_t i = size_t(*px++);
			if (i>=nca) RuntimeError("DoGatherColumnsOf; Map id=%lu is out of range [,%lu]", i, nca);

			size_t ns = compPos[i], ne = compPos[i+1];
			for (size_t k=ns; k<ne; ++k) PutItem(compId[k], j, pBuffer[k]);
		}
	}
	else if (a.GetItemCount()==0 || alpha==0)
	{
		ElemType* pBuffer = GetBuffer();
		index_t* compPos = GetPrimePos();
		index_t* compId = GetCompId();
		for (size_t j=0; j<m_numCols; ++j)
		{
			if (std::isnan(*px) || *px < 0) { ++px; continue; }
			size_t i = size_t(*px++);
			if (i>=nca) RuntimeError("DoGatherColumnsOf; Map id=%lu is out of range [,%lu]", i, nca);

			size_t ns = compPos[i], ne = compPos[i+1];
			for (size_t k=ns; k<ne; ++k) pBuffer[k] *= beta;
		}
	}
	else
	{
		CPUSparseMatrix<ElemType> sm(m_numRows, m_numCols, GetItemCount(), GetFormat());
		ElemType* pa = new ElemType[2*m_numRows];
		ElemType* pc = pa + m_numRows;
		const index_t* compPos1 = a.GetPrimePos();
		const index_t* compPos2 = GetPrimePos();
		for (size_t j=0; j<m_numCols; ++j,++px)
		{
			int upd = 0;
			if (compPos2[j]<compPos2[j+1]) { GetData(pc, j); ++upd; }
			else memset(pc, 0, m_numRows*sizeof(ElemType));

			if (!std::isnan(*px) && *px >= 0)
			{
				size_t i = size_t(*px);
				if (i>=nca) RuntimeError("DoGatherColumnsOf; Map id=%lu is out of range [,%lu]", i, nca);

				// version 1
				//a.GetData(pa, i); ++upd;
				//for (size_t i=0; i<m_numRows; ++i) pc[i] = alpha*pa[i] + beta*pc[i];

				// version 2
				if (compPos1[i]<compPos1[i+1]) { a.GetData(pa, i); ++upd; }
				else memset(pa, 0, m_numRows*sizeof(ElemType));
				if (upd) for (size_t i=0; i<m_numRows; ++i) pc[i] = alpha*pa[i] + beta*pc[i];
			}
			if (upd) sm.PutData(pc, j);
		}
		Assign(sm, true);
		delete[] pa;
	}
	return *this;
}

// *this[:,idx[j]] = a[:,j] * alpha + *this[:,idx[j]] * beta
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
///void CPUSparseMatrix<ElemType>::SetMatrixFromBSCFormat(const index_t* blockPos, const ElemType* pVal, size_t numBlocks, size_t numRows, size_t numCols)
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

// c = alpha * a * b + beta * c
// dense * sparse -> dense
template <class ElemType>
void CPUSparseMatrix<ElemType>::MultiplyAndWeightedAdd(ElemType alpha, const CPUMatrix<ElemType>& a, bool transposeA,
																const CPUSparseMatrix<ElemType>& b, bool transposeB,
																ElemType beta, CPUMatrix<ElemType>& c)
{
	if (a.IsEmpty() || b.IsEmpty())
		LogicError("MultiplyAndWeightedAdd; A or B matrix is empty");
	if (b.GetFormat() != matrixFormatSparseCSC)
		LogicError("MultiplyAndWeightedAdd; B matrix is not SparseCSC");

	size_t anr = (transposeA) ? a.GetNumCols() : a.GetNumRows();
	size_t anc = (transposeA) ? a.GetNumRows() : a.GetNumCols();
	size_t bnr = (transposeB) ? b.GetNumCols() : b.GetNumRows();
	size_t bnc = (transposeB) ? b.GetNumRows() : b.GetNumCols();

	// A(k,m) * B(m,n) --> C(k,n)
	if (anc != bnr)
		InvalidArgument("MultiplyAndWeightedAdd; The inner dimensions of a (%lu) and b (%lu) don't match", anc, bnr);

	if (c.HasExternalBuffer())
		LogicError("MultiplyAndWeightedAdd; Cannot modify external buffer in matrix");

	if (c.IsEmpty()) c.Resize(anr,bnc);
	else if (anr!=c.GetNumRows() || bnc!=c.GetNumCols())
		InvalidArgument("MultiplyAndWeightedAdd; Different dimensions of a(,%lu) or b(%lu,) with c(%lu,%lu)", anc, bnr, c.GetNumRows(), c.GetNumCols());
	else if (beta==0) c.Reset();
	else if (beta!=1) c *= beta;
	//c.ConvertToFullBlock();

	ElemType* pdc = c.GetData();
	const ElemType* pda = a.GetBuffer();
	const ElemType* pdb = b.GetBuffer();
	const index_t* compPos = b.GetPrimePos();
	const index_t* compId = b.GetCompId();

	anr = a.GetNumRows(); anc = a.GetNumCols();
	bnr = b.GetNumRows(); bnc = b.GetNumCols();

	int m = (transposeA ? 2:0) + (transposeB ? 1:0);
	if (m==0)
	{
		// C = alpha * A * B + beta * C
		for (size_t j=0; j<bnc; ++j)
		{
			size_t ns = compPos[j], ne = compPos[j+1]; if (ns==ne) continue;
			ElemType* pc = pdc + j*anr;
			for (size_t n=ns; n<ne; ++n)
			{
				ElemType val = alpha*pdb[n];
				const ElemType* pa = pda + compId[n]*anr;
				for (size_t k=0; k<anr; ++k) pc[k] += val*pa[k];
			}
		}
	}
	else if (m==1)
	{
		// C = alpha * A * Bt + beta * C
		for (size_t j=0; j<bnc; ++j)
		{
			size_t ns = compPos[j], ne = compPos[j+1]; if (ns==ne) continue;
			const ElemType* pa = pda + j*anr;
			for (size_t n=ns; n<ne; ++n)
			{
				ElemType val = alpha*pdb[n];
				ElemType* pc = pdc + compId[n]*anr;
				for (size_t k=0; k<anr; ++k) pc[k] += val*pa[k];
			}
		}
	}
	else if (m==2)
	{
		// C = alpha * At * B + beta * C
		for (size_t j=0; j<bnc; ++j)
		{
			size_t ns = compPos[j], ne = compPos[j+1]; if (ns==ne) continue;
			for (size_t n=ns; n<ne; ++n)
			{
				ElemType val = alpha*pdb[n];
				ElemType* pc = pdc + j*anc;
				const ElemType* pa = pda + compId[n];
				for (size_t k=0; k<anc; ++k) { pc[k] += val*(*pa); pa += anr; }
			}
		}
	}
	else
	{
		// C = alpha * At * Bt + beta * C
		for (size_t j=0; j<bnc; ++j)
		{
			size_t ns = compPos[j], ne = compPos[j+1]; if (ns==ne) continue;
			for (size_t n=ns; n<ne; ++n)
			{
				ElemType val = alpha*pdb[n];
				ElemType* pc = pdc + compId[n]*anc;
				const ElemType* pa = pda + j;
				for (size_t k=0; k<anc; ++k) { pc[k] += val*(*pa); pa += anr; }
			}
		}
	}
}

// c = alpha * a * b + beta * c
// sparse * dense -> dense
template <class ElemType>
void CPUSparseMatrix<ElemType>::MultiplyAndWeightedAdd(ElemType alpha, const CPUSparseMatrix<ElemType>& a, bool transposeA,
																const CPUMatrix<ElemType>& b, bool transposeB,
																ElemType beta, CPUMatrix<ElemType>& c)
{
	if (a.IsEmpty() || b.IsEmpty())
		LogicError("CPUSparseMatrix::MultiplyAndAdd; A or B matrix is empty");
	if (a.GetFormat() != matrixFormatSparseCSC)
		LogicError("CPUSparseMatrix::MultiplyAndAdd; A matrix is not SparseCSC");

	size_t anr = (transposeA) ? a.GetNumCols() : a.GetNumRows();
	size_t anc = (transposeA) ? a.GetNumRows() : a.GetNumCols();
	size_t bnr = (transposeB) ? b.GetNumCols() : b.GetNumRows();
	size_t bnc = (transposeB) ? b.GetNumRows() : b.GetNumCols();

	if (c.IsEmpty()) c.Resize(anr,bnc);
	else if (anr!=c.GetNumRows() || bnc!=c.GetNumCols())
		InvalidArgument("MultiplyAndWeightedAdd; Different dimensions of a(,%lu) or b(%lu,) with c(%lu,%lu)", anc, bnr, c.GetNumRows(), c.GetNumCols());
	else if (beta==0) c.Reset();
	else if (beta!=1) c *= beta;
	//c.ConvertToFullBlock();

	ElemType* pdc = c.GetData();
	const ElemType* pda = a.GetBuffer();
	const ElemType* pdb = b.GetBuffer();
	const index_t* compPos = a.GetPrimePos();
	const index_t* compId = a.GetCompId();

	anr = a.GetNumRows(); anc = a.GetNumCols();
	bnr = b.GetNumRows(); bnc = b.GetNumCols();

	int m = (transposeA ? 2:0) + (transposeB ? 1:0);
	if (m==0)
	{
		// C += a * A * B
		for (size_t j=0; j<anc; ++j)
		{
			size_t ns = compPos[j], ne = compPos[j+1]; if (ns==ne) continue;
			for (size_t n=ns; n<ne; ++n)
			{
				size_t i = compId[n];
				ElemType val = alpha*pda[n];
				ElemType* pc = pdc + i;
				const ElemType* pb = pdb + j;
				for (size_t k=0; k<bnc; ++k) { *pc += val*(*pb); pc += anr; pb += bnr; }
			}
		}
	}
	else if (m==1)
	{
		// C += a * A * Bt
		for (size_t j=0; j<anc; ++j)
		{
			size_t ns = compPos[j], ne = compPos[j+1]; if (ns==ne) continue;
			for (size_t n=ns; n<ne; ++n)
			{
				size_t i = compId[n];
				ElemType val = alpha*pda[n];
				ElemType* pc = pdc + i;
				const ElemType* pb = pdb + j*bnr;
				for (size_t k=0; k<bnr; ++k) { *pc += val*(*pb++); pc += anr; }
			}
		}
	}
	else if (m==2)
	{
		// C += a * At * B
		for (size_t j=0; j<anc; ++j)
		{
			size_t ns = compPos[j], ne = compPos[j+1]; if (ns==ne) continue;
			for (size_t n=ns; n<ne; ++n)
			{
				size_t i = compId[n];
				ElemType val = alpha*pda[n];
				ElemType* pc = pdc + j;
				const ElemType* pb = pdb + i;
				for (size_t k=0; k<bnc; ++k) { *pc += val*(*pb); pc += anc; pb += bnr; }
			}
		}
	}
	else
	{
		// C += a * At * Bt
		for (size_t j=0; j<anc; ++j)
		{
			size_t ns = compPos[j], ne = compPos[j+1]; if (ns==ne) continue;
			for (size_t n=ns; n<ne; ++n)
			{
				size_t i = compId[n];
				ElemType val = alpha*pda[n];
				ElemType* pc = pdc + j;
				const ElemType* pb = pdb + i*bnr;
				for (size_t k=0; k<bnr; ++k) { *pc += val*(*pb++); pc += anc; }
			}
		}
	}
}

// c += alpha * lhs * rhs
// dense += alpha * dense * sparse
template <class ElemType>
void CPUSparseMatrix<ElemType>::MultiplyAndAdd(const CPUMatrix<ElemType>& a, bool transposeA, const CPUSparseMatrix<ElemType>& b, bool transposeB, CPUMatrix<ElemType>& c)
{
	MultiplyAndWeightedAdd(1, a, transposeA, b, transposeB, 1, c);
}

// c += alpha * lhs * rhs
// dense += alpha * dense * sparse
template <class ElemType>
void CPUSparseMatrix<ElemType>::MultiplyAndAdd(const CPUSparseMatrix<ElemType>& a, bool transposeA, const CPUMatrix<ElemType>& b, bool transposeB, CPUMatrix<ElemType>& c)
{
	MultiplyAndWeightedAdd(1, a, transposeA, b, transposeB, 1, c);
}

//	c[:,j] = alpha * v[j] * a[:,j] + beta * c[:,j]
//	dense = alpha * vector * sparse + beta * dense
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
///		const index_t* blockPos = a.GetBlockPos();
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
///		index_t* blockPos = rhs.GetBlockPos();
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
///		index_t* blockPos = lhs.GetBlockPos();
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

template class CPUSparseMatrix<half>;
template class CPUSparseMatrix<float>;
template class CPUSparseMatrix<double>;


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

