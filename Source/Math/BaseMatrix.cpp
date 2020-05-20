//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "BaseMatrix.h"

namespace Microsoft { namespace MSR { namespace CNTK {

//==============================================================================
//			TracingGPUMemoryAllocator
//==============================================================================

//int TracingGPUMemoryAllocator::m_traceLevel = 0;
//
//void TracingGPUMemoryAllocator::SetTraceLevel(int traceLevel) { m_traceLevel = traceLevel; }
//bool TracingGPUMemoryAllocator::IsTraceEnabled() { return (m_traceLevel > 0); }

//==============================================================================
//			BaseMatrix
//==============================================================================

template<class ElemType>
void BaseMatrix<ElemType>::Init(MatrixFormat fmt, device_t dev)
{
	m_numRows = m_numCols = m_sliceOffset = 0;
	if (m_sob==nullptr) m_sob = make_shared<BaseMatrixStorage<ElemType>>(fmt, dev);
	else m_sob->Init(fmt, dev);
}

template<class ElemType>
void BaseMatrix<ElemType>::Assign(size_t rows, size_t cols, ElemType* p, int flags)
{
	m_sob->Create(m_numRows=rows, m_numCols=cols, p, flags);
	m_sliceOffset = 0;
}

template<class ElemType>
void BaseMatrix<ElemType>::Assign(const BaseMatrix<ElemType>& mat, bool shallow)
{
	if (&mat == this) return;
	m_numRows = mat.m_numRows;
	m_numCols = mat.m_numCols;
	m_sliceOffset = mat.m_sliceOffset;
	if (shallow) m_sob = mat.m_sob;
	else m_sob->Assign(*mat.m_sob.get());
}

template<class ElemType>
void BaseMatrix<ElemType>::Resize(size_t rows, size_t cols)
{
	m_sliceOffset = 0;
	m_numRows = rows; m_numCols = cols;
	if (m_sob==nullptr) m_sob = make_shared<BaseMatrixStorage<ElemType>>(matrixFormatDense, CPUDEVICE);
	if (rows!=m_sob->GetNumRows() || cols!=m_sob->GetNumCols()) m_sob->Create(rows, cols);
	else m_sob->SetZeros();
}

template<class ElemType>
void BaseMatrix<ElemType>::Reshape(size_t rows, size_t cols)
{
	if (m_sob->Reshape(rows,cols)) ResizeBack();
	else LogicError("Reshape; Invalid new shape");
}

template<class ElemType>
void BaseMatrix<ElemType>::SetSlice(size_t start, size_t len)
{
	MatrixFormat mft = m_sob->GetFormat();
	if (mft & matrixFormatRowMajor)
	{
		if (start+len > m_numRows)
			RuntimeError("SetSlice (%lu..%lu) is out of range=%lu", start, start+len-1, m_numRows);
		if (mft & matrixFormatSparse) m_sliceOffset += start;
		else m_sliceOffset += start * m_numCols;
		m_numRows = len;
	}
	else
	{
		if (start+len > m_numCols)
			RuntimeError("SetSlice (%lu..%lu) is out of range=%lu", start, start+len-1, m_numCols);
		if (mft & matrixFormatSparse) m_sliceOffset += start;
		else m_sliceOffset += start * m_numRows;
		m_numCols = len;
	}
}

template<class ElemType>
void BaseMatrix<ElemType>::SetColumnSlice(size_t start, size_t len)
{
	MatrixFormat mft = m_sob->GetFormat();
	if (mft & matrixFormatRowMajor) RuntimeError("Coulmn slice is not supported for row major matrix");
	if (start+len > m_numCols) RuntimeError("Column slice (%lu..%lu) is out of range=%lu", start, start+len, m_numCols);
	if (mft & matrixFormatSparse) m_sliceOffset += start;
	else m_sliceOffset += start * m_numRows;
	m_numCols = len;
}

//template<class ElemType>
//size_t BaseMatrix<ElemType>::NzCount() const
//{
//	size_t n = m_numRows*m_numCols; if (n==0) return 0;
//
//	// dense
//	MatrixFormat mft = m_sob->GetFormat();
//	if ((mft & matrixFormatSparse)==0) return n;
//
//	size_t nc = (mft & matrixFormatRowMajor) ? m_numRows : m_numCols;
//	size_t nr = (mft & matrixFormatRowMajor) ? m_numCols : m_numRows;
//	if ((mft & matrixFormatBlock)==0)
//	{
//		// sparse
//		const index_t* primePos = m_sob->GetCompPos() + m_sliceOffset;
//		return primePos[nc] - primePos[0];
//	}
//	const size_t* blockPos = m_sob->GetBlockPos();
//	if (blockPos)
//	{
//		// block (new)
//		size_t n = 0; blockPos += m_sliceOffset;
//		for (size_t j=0; j<nc; ++j) if (blockPos[j]!=string::npos) ++n;
//		return n*nr;
//	}
//	const size_t* blockId = m_sob->GetBlockId();
//	if (blockId)
//	{
//		// block (old)
//		size_t n = 0, nblk = m_sob->GetBlockCount();
//		size_t ns = m_sliceOffset, ne = m_sliceOffset + nc;
//		for (size_t j=0; j<nblk; ++j) if (j>=ns && j<ne) ++n;
//		return n*nr;
//	}
//	return 0;
//}

template<class ElemType>
void BaseMatrix<ElemType>::CopyToDense(BaseMatrix<ElemType>& mat) const
{
	if (&mat == this) return;
	size_t nc = (IsRowMajor()) ? m_numRows : m_numCols;
	mat.m_sob->MakeDenseFrom(*m_sob.get(), m_sliceOffset, nc);
	mat.m_numRows = m_numRows;
	mat.m_numCols = m_numCols;
	mat.m_sliceOffset = 0;
}

template<class ElemType>
void BaseMatrix<ElemType>::CopyToSparse(BaseMatrix<ElemType>& mat) const
{
	if (&mat == this) return;
	size_t nc = (IsRowMajor()) ? m_numRows : m_numCols;
	mat.m_sob->MakeSparseFrom(*m_sob.get(), m_sliceOffset, nc);
	mat.m_numRows = m_numRows;
	mat.m_numCols = m_numCols;
	mat.m_sliceOffset = 0;
}

template<class ElemType>
void BaseMatrix<ElemType>::CopyToBlock(BaseMatrix<ElemType>& mat) const
{
	if (&mat == this) return;
	size_t nc = (IsRowMajor()) ? m_numRows : m_numCols;
	mat.m_sob->MakeBlockFrom(*m_sob.get(), m_sliceOffset, nc);
	mat.m_numRows = m_numRows;
	mat.m_numCols = m_numCols;
	mat.m_sliceOffset = 0;
}

template<class ElemType>
void BaseMatrix<ElemType>::CopyToFullBlock(BaseMatrix<ElemType>& mat) const
{
	if (&mat == this) return;
	size_t nc = (IsRowMajor()) ? m_numRows : m_numCols;
	mat.m_sob->MakeFullBlockFrom(*m_sob.get(), m_sliceOffset, nc);
	mat.m_numRows = m_numRows;
	mat.m_numCols = m_numCols;
	mat.m_sliceOffset = 0;
}

template<class ElemType>
bool BaseMatrix<ElemType>::IsEqualTo(const BaseMatrix<ElemType>& m, ElemType thresh) const
{
	if (m.m_numRows!=m_numRows || m.m_numCols!=m_numCols) return false;
	for (size_t j=0; j<m_numCols; ++j)
	for (size_t i=0; i<m_numRows; ++i)
		if (abs(GetItem(i,j)-m.GetItem(i,j)) > thresh)
		{
			float x = float(GetItem(i,j)), y = float(m.GetItem(i,j));
			cout << "*** (" << i << "," << j << ") " << x << " / " << y << "  d=" << abs(x-y) << endl;
			return false;
		}
	return true;
}

template<class ElemType>
int BaseMatrix<ElemType>::Compare(const BaseMatrix<ElemType>& mat) const
{
	if (mat.m_numRows!=m_numRows) return diffRows;
	if (mat.m_numCols!=m_numCols) return diffCols;
	if (mat.m_sliceOffset!=m_sliceOffset) return diffSlice;
	int m = (m_sob==nullptr ? 0:1) + (mat.m_sob==nullptr ? 0:2);
	if (m==1 || m==2) return diffStorage;
	return (m==0 ? 0 : m_sob->Compare(*mat.m_sob.get()));
}

template<class ElemType>
void BaseMatrix<ElemType>::ViewData(ostream& os, const char* fmt) const
{
	if (m_numRows*m_numCols > 0)
		m_sob->ViewBuffer(os, fmt, m_sliceOffset, (m_sob->IsRowMajor() ? m_numRows : m_numCols));
}

template class BaseMatrix<char>;
template class BaseMatrix<short>;
template class BaseMatrix<int>;
template class BaseMatrix<half>;
template class BaseMatrix<float>;
template class BaseMatrix<double>;

} } }
