//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "ExportAPI.h"
#include "../Common/Basics.h"
#include "../Common/BaseTypes.h"
#include "half.hpp"
//#include <string>
//#include <stdint.h>
//#include <memory>
//#include <unordered_map>
//#include <map>

#pragma warning( disable: 4251 )
typedef unsigned char byte;

#define device_t	int
#define index_t		int

// and the following magic values
#define CPUDEVICE			-1			// device is the CPU
#define DEVICE_NOTSET		-3			// not yet set
#define DEVICE_AUTO			-4			// device should be picked automatically

#define EPS_IN_INVERSE		1e-30f		// 1e-37 is the only guaranteed precision
#define EPS_IN_LOG			1e-37f		// 1e-37 is the only guaranteed precision
#define LOG_EPS_IN_LOG		-85.1f		// log(EPS_IN_LOG)
#define LOG10_EPS_IN_LOG	-37			// log_10(EPS_IN_LOG)
#define LZERO				-10e10
#define MINLOGEXP			-9.2103
#define LSMALL				-0.5E10

// special markers in BlockId2ColOrRow()/ColOrRow2BlockId()
//static const index_t SparseIndex_NotAssigned = -1; // the index is not used, for col2BlockId it means the column has no corresponding block
//static const index_t SparseIndex_Pending = -2; // the index assignment is pending, a transitional state when counting new blocks when sparse += sparse * dense 

namespace Microsoft { namespace MSR { namespace CNTK {

///MATH_API void SetMathLibTraceLevel(int traceLevel);
///MATH_API int  GetMathLibTraceLevel();

inline bool IsCpu(device_t dev) { return dev < 0; }
inline bool IsGpu(device_t dev) { return dev >= 0; }

inline size_t buffsize(size_t n, size_t m=2) { return (n + (m-1)) & ~(m-1); }

//==============================================================================
//		TracingGPUMemoryAllocator
//==============================================================================

class MATH_API TracingGPUMemoryAllocator
{
private:
	static int m_traceLevel;

public:
	static void SetTraceLevel(int traceLevel);
	static bool IsTraceEnabled();

	template <typename AllocatedElemType>
	static AllocatedElemType* Allocate(device_t dev, size_t numRows, size_t numCols);

	template <typename AllocatedElemType>
	static AllocatedElemType* Allocate(device_t dev, size_t numElements);

	template <typename AllocatedElemType>
	static void Free(device_t dev, AllocatedElemType* bufferPtr, bool ignoreCUDARetCode = false);

	// Let it be public method, the memory manager could check the totoal free memory and
	// decide whether to physically release all the cached memory.
	static std::pair<size_t, size_t> GetFreeAndTotalMemoryInMBs(device_t dev);

private:
	template <typename AllocatedElemType>
	static AllocatedElemType* AllocateNoTrace(device_t dev, size_t numElements);
};

//==============================================================================
//		ElementWiseOperator
//		This enum represents which function to apply.
//		This is shared between all matrix types and tensors.
//==============================================================================

enum ElementWiseOperator
{
    // nullary
    opConstOne, opNone,
    // unary (or binary with constant parameter)
    opCopy,
    opNegate, opNot, opAbs, opFloor, opReciprocal,
    opSigmoid, opTanh, opAtanh, opSqr, opSqrt, opExp, opLog, opLinearRectifier,
    opCosine, opSin, opTan, opAcos, opAsin, opAtan, opCosh, opSinh, opAsinh, opExponentialLinearUnit, opStableSigmoid, opStraightThrough,
    // unary ops for use by Matrix class only (there is no TensorView implementation)
    opSigmoidDerivative, opLinearRectifierDerivative, opNegativeSine, opExponentialLinearUnitDerivative, opStableSigmoidDerivative, opStraightThroughDerivative,
    // binary
    opCopyIf, opCopyIfNot, opSum, opDifference, opElementwiseProduct, opElementwiseQuotient, opLogSum, opPow,
    opMax, opMin, opArgmax, opArgmin,
    opLess, opEqual, opGreater, opGreaterEqual, opNotEqual, opLessEqual, // Note: must obey this order: (sgn(a-b) == -1, 0, +1), (sgn(a-b) != -1, 0, +1)
    opAnd, opOr, opXor, opMaskNegative,
    opElementwiseProductWithSigmoidDerivativeFromOutput, opElementwiseProductWithTanhDerivativeFromOutput,
    opElementwiseProductWithLinearRectifierDerivativeFromOutput, opElementwiseProductWithLogDerivativeFromOutput,
    opElementwiseProductWithCosDerivative, opElementwiseProductWithSinDerivative, opElementwiseProductWithTanDerivative,
    opElementwiseProductWithAcosDerivative, opElementwiseProductWithAsinDerivative, opElementwiseProductWithAtanDerivative,
    opElementwiseProductWithCoshDerivative, opElementwiseProductWithSinhDerivative,
    opElementwiseProductWithAtanhDerivative, opElementwiseProductWithAsinhDerivative,
    opElementwiseProductWithAbsDerivative, opElementwiseProductWithSqrtDerivative,
    opElementwiseProductWithReciprocalDerivative, opSqrOfDifference,
    opElementwiseProductWithExponentialLinearUnitDerivativeFromOutput,
    opElementwiseProductWithStraightThroughDerivative,
    // binary ops for indexing
    // opIndex,
    // ternary
    opCond /*a ? b : c*/,
    opClip, /*clip a within interval b..c*/
    opElementwiseProductWithLogSumDerivative,
    opCopyIfEqual,
    opElementwiseProductWithExpOfDiff, /* a * exp(b - c) */
    opElementwiseProductWithQuotient, /* a * (b / c) */
    opElementwiseProductWithPowExponentDerivative, /* a * b * log(c) */
    opElementwiseProductWithPowBaseDerivative,  /* a * c * pow(b, c-1) */
    // Note: not all that's implemented in CNTK ComputationNodes has an opcode yet.
};

// helper to apply a C macro for all operations of each kind
#define ForAllNullaryOps(Macro) \
    Macro(ConstOne);

#define ForAllUnaryOps(Macro)     \
    Macro(Copy);                  \
    Macro(Negate);                \
    Macro(Not);                   \
    Macro(Abs);                   \
    Macro(Floor);                 \
    Macro(Reciprocal);            \
    Macro(Sigmoid);               \
    Macro(Tanh);                  \
    Macro(Atanh);                 \
    Macro(Sqr);                   \
    Macro(Sqrt);                  \
    Macro(Exp);                   \
    Macro(Log);                   \
    Macro(LinearRectifier);       \
    Macro(Cosine);                \
    Macro(Sin);                   \
    Macro(Tan);                   \
    Macro(Acos);                  \
    Macro(Asin);                  \
    Macro(Atan);                  \
    Macro(Cosh);                  \
    Macro(Sinh);                  \
    Macro(Asinh);                 \
    Macro(ExponentialLinearUnit); \
    Macro(StableSigmoid);         \
    Macro(StraightThrough);

#define ForAllBinaryOps(Macro)                                               \
    Macro(CopyIf);                                                           \
    Macro(CopyIfNot);                                                        \
    Macro(Sum);                                                              \
    Macro(Difference);                                                       \
    Macro(ElementwiseProduct);                                               \
    Macro(ElementwiseQuotient);                                              \
    Macro(LogSum);                                                           \
    Macro(Pow);                                                              \
    Macro(Max);                                                              \
    Macro(Min);                                                              \
    Macro(Equal);                                                            \
    Macro(NotEqual);                                                         \
    Macro(Greater);                                                          \
    Macro(Less);                                                             \
    Macro(GreaterEqual);                                                     \
    Macro(LessEqual);                                                        \
    Macro(And);                                                              \
    Macro(Or);                                                               \
    Macro(Xor);                                                              \
    Macro(MaskNegative);                                                     \
    Macro(ElementwiseProductWithSigmoidDerivativeFromOutput);                \
    Macro(ElementwiseProductWithTanhDerivativeFromOutput);                   \
    Macro(ElementwiseProductWithAtanhDerivative);                            \
    Macro(ElementwiseProductWithLinearRectifierDerivativeFromOutput);        \
    Macro(ElementwiseProductWithLogDerivativeFromOutput);                    \
    Macro(ElementwiseProductWithCosDerivative);                              \
    Macro(ElementwiseProductWithSinDerivative);                              \
    Macro(ElementwiseProductWithTanDerivative);                              \
    Macro(ElementwiseProductWithAcosDerivative);                             \
    Macro(ElementwiseProductWithAsinDerivative);                             \
    Macro(ElementwiseProductWithAtanDerivative);                             \
    Macro(ElementwiseProductWithCoshDerivative);                             \
    Macro(ElementwiseProductWithSinhDerivative);                             \
    Macro(ElementwiseProductWithAsinhDerivative);                            \
    Macro(ElementwiseProductWithAbsDerivative);                              \
    Macro(ElementwiseProductWithReciprocalDerivative);                       \
    Macro(ElementwiseProductWithSqrtDerivative);                             \
    Macro(SqrOfDifference);                                                  \
    Macro(ElementwiseProductWithExponentialLinearUnitDerivativeFromOutput);  \
    Macro(ElementwiseProductWithStraightThroughDerivative); 
    //Macro(Index);

#define ForAllTernaryOps(Macro)                         \
    Macro(Cond);                                        \
    Macro(CopyIfEqual);                                 \
    Macro(Clip);                                        \
    Macro(ElementwiseProductWithLogSumDerivative);      \
    Macro(ElementwiseProductWithExpOfDiff);             \
    Macro(ElementwiseProductWithQuotient);              \
    Macro(ElementwiseProductWithPowExponentDerivative); \
    Macro(ElementwiseProductWithPowBaseDerivative);

//==============================================================================
//		various enums to describe
//==============================================================================

///enum class MatrixOrder
///{
///    RowMajor = 101,
///    ColMajor = 102
///};
///
///enum class MatrixTranspose : char
///{
///    NoTrans = 'N',
///    Trans = 'T',
///    ConjTrans = 'C'
///};
///
///enum class SymMatrixType : char
///{
///    Up = 'U',				// symmetric matrix is stored in the upper part
///    Low = 'L',				// symmetric matrix is stored in the lower part
///    Full = 'F',				// full populated
///    NotSymmetric = 'N'		// not a symmetric matrix
///};
///
///enum class MatrixOpSide : char
///{
///    Left = 'L',				// left multiply
///    Right = 'R',			// right multiply
///};

enum MatrixFormat : int
{
	matrixFormatDense			= 0,			// default is dense
	matrixFormatColMajor		= 0,			// default is column major
	matrixFormatRowMajor		= 0x0001,		// row major matrix
	matrixFormatSparse			= 0x0002,		// sparse matrix
	matrixFormatBlock			= 0x0004,

	matrixFormatDenseCol		= matrixFormatDense + matrixFormatColMajor,
	matrixFormatDenseRow		= matrixFormatDense + matrixFormatRowMajor,
	matrixFormatSparseCSC		= matrixFormatSparse + matrixFormatColMajor,
	matrixFormatSparseCSR		= matrixFormatSparse + matrixFormatRowMajor,
	matrixFormatSparseBSC		= matrixFormatSparse + matrixFormatColMajor + matrixFormatBlock,
	matrixFormatSparseBSR		= matrixFormatSparse + matrixFormatRowMajor + matrixFormatBlock,

	matrixFormatMask			= matrixFormatSparse + matrixFormatRowMajor + matrixFormatBlock,
	matrixFormatSparseBlock		= matrixFormatSparse + matrixFormatBlock,

	matrixFlagNone				= 0,
	matrixFlagExternalBuffer	= 0x0010,		// the memory pointers are externally managed
	matrixFlagSetValueOnDevice	= 0x0020,		// SetValue() call has a buffer that is already on the device
};

enum MatrixDiffrence : int
{
	diffNone,
	diffRows,
	diffCols,
	diffSlice,
	diffStorage,
	diffDevice,
	diffFormat,
	diffIndex,
	diffData
};

//==============================================================================
//		BaseMatrixStorage
//		base class for all matrix types (CPU, GPU) x (dense, sparse)
//==============================================================================

template <class ElemType>
class BaseMatrixStorage : public enable_shared_from_this<BaseMatrixStorage<ElemType>>
{
	template <class ElemType2> friend class BaseMatrix;

protected:
	// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
	//		Variables required by all matrices
	// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

	MatrixFormat	m_format;
	device_t		m_deviceId;

	size_t			m_numRows;
	size_t			m_numCols;
	size_t			m_allocSize;			// element size, buffer
	bool			m_extBuffer;
	ElemType*		m_pBuffer;

	// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
	//		CPUSparseMatrix variables
	// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

	index_t*		m_compPos;				// begin ids of col/row in CSC/CSR format
	index_t*		m_compId;				// row/col ids in CSC/CSR format
	size_t			m_compSize;

	size_t*			m_blockId;				// block ids
	size_t*			m_blockPos;
	size_t			m_blockCnt;				// how many blocks in matrix

	// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
	//		GPUSparseMatrix variables
	// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

	index_t*		m_tempDeviceBuffer;
	size_t			m_tempDeviceBufferSize;
	void*			m_tempHostBuffer;
	size_t			m_tempHostBufferSize;

private:
	BaseMatrixStorage<ElemType>(const BaseMatrixStorage<ElemType>&) = delete;
	BaseMatrixStorage<ElemType>& operator=(const BaseMatrixStorage<ElemType>&) = delete;
public:
	BaseMatrixStorage(MatrixFormat mft = matrixFormatDense, device_t dev = CPUDEVICE) { ZeroInit(mft, dev); }
	~BaseMatrixStorage() { Release(); }

	void Release();
	void Allocate(size_t n);
	void Init(MatrixFormat mft, device_t dev = CPUDEVICE) { Release(); m_format = mft; m_deviceId = dev; }
	void Reset();

	size_t Create(size_t rows, size_t cols);
	//void Create(size_t rows, size_t cols, ElemType* p, int flags);
	//void Assign(const BaseMatrixStorage<ElemType>& bms);

	//void MakeDenseFrom(const BaseMatrixStorage<ElemType>& bms);
	//void MakeSparseFrom(const BaseMatrixStorage<ElemType>& bms);
	//void MakeBlockFrom(const BaseMatrixStorage<ElemType>& bms);
	//void MakeFullBlock();

	MatrixFormat GetFormat() const { return m_format; }
	device_t GetDeviceId() const { return m_deviceId; }

	bool IsDenseFormat() const { return (m_format & matrixFormatSparseBlock)==0; }
	bool IsSparseFormat() const { return (m_format & matrixFormatSparseBlock)==matrixFormatSparse; }
	bool IsBlockFormat() const { return (m_format & matrixFormatSparseBlock)==matrixFormatSparseBlock; }
	bool IsColMajor() const { return (m_format & matrixFormatRowMajor)==0; }
	bool IsRowMajor() const { return (m_format & matrixFormatRowMajor)!=0; }

	size_t GetNumRows() const { return m_numRows; }
	size_t GetNumCols() const { return m_numCols; }
	size_t GetNumElements() const { return m_numRows * m_numCols; }
	//size_t GetSizeAllocated() const { return m_allocSize; }
	//size_t GetItemCount() const;

	bool IsEmpty() const { return m_numRows == 0 || m_numCols == 0; }
	bool HasExternalBuffer() const { return m_extBuffer; }

	ElemType* GetBuffer() const { return m_pBuffer; }

	//void SetFormat(MatrixFormat mft) { m_format = mft; }
	//void SetDeviceId(device_t dev) { m_deviceId = dev; }

	//void SetNumRows(size_t rows) { m_numRows = rows; }
	//void SetNumCols(size_t cols) { m_numCols = cols; }
	//void SetSizeAllocated(size_t alloc) { m_allocSize = alloc; }

	//void SetBuffer(ElemType* p, size_t total, bool external = false) { m_pBuffer = p; /*m_totalSize = total;*/ m_extBuffer = external; }

	//ElemType GetItem(size_t row, size_t col, size_t offset=0) const;
	//void PutItem(size_t row, size_t col, ElemType val, size_t offset=0);

	void SetZeros();
	//void Transpose() { size_t n = m_numRows; m_numRows = m_numCols; m_numCols = n; m_format = MatrixFormat(m_format ^ matrixFormatRowMajor); }
	//void TransposeFrom(const BaseMatrixStorage<ElemType>& bms, bool hdr);

	//int  Compare(const BaseMatrixStorage<ElemType>& bms) const;

	// CPU sparse format
	//size_t GetBlockCount() const { return m_blockCnt; }
	//size_t GetCompPosSize() const { return m_compSize; }

	index_t* GetCompPos() const { return m_compPos; }
	index_t* GetCompId() const { return m_compId; }

	size_t* GetBlockId() const { return m_blockId; }
	size_t* GetBlockPos() const { return m_blockPos; }

	//void SetBlockCount(size_t n) { m_blockCnt = n; }
	//void SetCompPosSize(size_t n) { m_compSize = n; }
	//void SetBlockIdShift(size_t n) { m_blockShift = n; }

	//void SetCompPos(index_t* p) { m_compPos = p; }
	//void SetCompId(index_t* p) { m_compId = p; }
	//void SetBlockId(size_t* p) { m_blockId = p; }
	//void SetNzValues(ElemType* p) { m_nzValues = p; }

	// GPU
	//void* GetTempHostBuffer() const { return m_tempHostBuffer; }
	//size_t GetTempHostBufferSize() const { return m_tempHostBufferSize; }

	//index_t* GetTempDeviceBuffer() const { return m_tempDeviceBuffer; }

	//void SetTempHostBuffer(void* buffer) { m_tempHostBuffer = buffer; }
	//void SetTempHostBufferSize(size_t bufferSize) { m_tempHostBufferSize = bufferSize; }
	//void ReserveTempDeviceBuffer(size_t minSize) const
	//{
//#ifndef CPUONLY
//		BaseMatrixStorage<ElemType>* nonConstThis = const_cast<BaseMatrixStorage<ElemType>*>(this);
//		if (minSize > m_tempDeviceBufferSize)
//		{
//			TracingGPUMemoryAllocator::Free<index_t>(GetDeviceId(), nonConstThis->m_tempDeviceBuffer);
//			nonConstThis->m_tempDeviceBuffer = TracingGPUMemoryAllocator::Allocate<index_t>(GetDeviceId(), minSize);
//			nonConstThis->m_tempDeviceBufferSize = minSize;
//		}
//#endif
//	}
	// view
	string Format() const;
	string GetInfo(bool all=true) const;
	void ViewBuffer(ostream& os, const char* fmt = "%8.4f", size_t pos=0, size_t len=0) const;
	void ViewIds(ostream& os) const;

protected:
	void ZeroInit(MatrixFormat mft, device_t dev);

	//void GetBlockId(vector<size_t>& v) const;

	void ViewDense(ostream& os, const char* fmt, size_t pos, size_t len) const;
	void ViewSparse(ostream& os, const char* fmt, size_t pos, size_t len) const;
	void ViewBlock1(ostream& os, const char* fmt, size_t pos, size_t len) const;
	void ViewBlock2(ostream& os, const char* fmt, size_t pos, size_t len) const;
};

template<class ElemType>
void BaseMatrixStorage<ElemType>::ZeroInit(MatrixFormat mft, device_t dev)
{
	m_format = mft;
	m_deviceId = dev;
	m_numRows = 0;
	m_numCols = 0;
	m_allocSize = 0;
	m_pBuffer = nullptr;
	m_extBuffer = false;

	// sparse matrix
	m_compSize = 0;					// sparse compressed
	m_compPos = nullptr;			// compressed slice position
	m_compId = nullptr;				// data Id
	m_blockCnt = 0;				// sparse block	counter
	m_blockId = nullptr;
	m_blockPos = nullptr;

	// GPU support
	m_tempDeviceBuffer = nullptr;
	m_tempDeviceBufferSize = 0;
	m_tempHostBuffer = nullptr;		// used to copy values.
	m_tempHostBufferSize = 0;
}

template<class ElemType>
void BaseMatrixStorage<ElemType>::Release()
{
	if (!m_extBuffer)
	{
		if (m_deviceId < 0)
		{
			delete[] m_pBuffer; m_pBuffer = nullptr;
			delete[] m_compPos; m_compPos = nullptr;
			delete[] m_compId; m_compId = nullptr;
			delete[] m_blockId; m_blockId = nullptr;
			delete[] m_blockPos; m_blockPos = nullptr;
		}
		else
		{
#ifndef CPUONLY
			if (m_pBuffer != nullptr)
				TracingGPUMemoryAllocator::Free<ElemType>(m_deviceId, m_pBuffer, true);
			m_pBuffer = nullptr;

			if (m_tempDeviceBuffer != nullptr)
				TracingGPUMemoryAllocator::Free<index_t>(m_deviceId, m_tempDeviceBuffer, true);
			m_tempDeviceBuffer = nullptr;
			m_tempDeviceBufferSize = 0;
#endif
			delete[](byte*) m_tempHostBuffer; m_tempHostBuffer = nullptr;
		}
		m_compSize = m_blockCnt = m_allocSize = 0;
		m_numRows = m_numCols = 0;
	}
}

template<class ElemType>
void BaseMatrixStorage<ElemType>::Allocate(size_t n)
{
	size_t k = buffsize(n); if (k<=m_allocSize) return;
	ElemType* pbuff = m_pBuffer; m_pBuffer = new ElemType[k];
	if (m_allocSize>0) memcpy(m_pBuffer, pbuff, m_allocSize*sizeof(ElemType));
	memset(m_pBuffer+m_allocSize, 0, (k-m_allocSize)*sizeof(ElemType));
	delete[] pbuff;

	if (IsSparseFormat())
	{
		index_t* pid = m_compId; m_compId = new index_t[k];
		if (m_allocSize>0) memcpy(m_compId, pid, m_allocSize*sizeof(index_t));
		delete [] pid;
	}
	m_allocSize = k;
}

template<class ElemType>
void BaseMatrixStorage<ElemType>::Reset()
{
	if (IsDenseFormat()) { size_t n = m_numRows*m_numCols; if (n) memset(m_pBuffer, 0, n*sizeof(ElemType)); }
	else if (IsSparseFormat()) memset(m_compPos, 0, m_compSize*sizeof(index_t));
	else if (m_blockPos) memset(m_blockPos, 0xff, m_compSize*sizeof(size_t));
	m_blockCnt = 0;
}

template<class ElemType>
size_t BaseMatrixStorage<ElemType>::Create(size_t rows, size_t cols)
{
	size_t n = rows * cols;
	m_numRows = rows; m_numCols = cols;
	if (IsDenseFormat())
	{
		if (m_extBuffer) { m_pBuffer = nullptr; m_extBuffer = false; }
		if (n>0) Allocate(n);
	}
	else if (IsSparseFormat())
	{
		size_t k = max(rows,cols) + 1;
		if (k > m_compSize) { delete[] m_compPos; m_compPos = new index_t[m_compSize=k]; }
		memset(m_compPos, 0, m_compSize*sizeof(index_t));
	}
	else if (IsBlockFormat())
	{
		size_t k = max(rows,cols) + 1;
		if (k > m_compSize) { delete[] m_blockPos; m_blockPos = new size_t[m_compSize=k]; }
		memset(m_blockPos, 0xff, m_compSize*sizeof(size_t));
		m_blockCnt = 0;
	}
	return n;
}

///template<class ElemType>
///void BaseMatrixStorage<ElemType>::Create(size_t rows, size_t cols, ElemType* p, int flags)
///{
///	if (IsDenseFormat())
///	{
///		// dense
///		if (flags & matrixFlagExternalBuffer)
///		{
///			m_numRows = rows; m_numCols = cols;
///			if (!m_extBuffer && m_allocSize>0) delete[] m_pBuffer;
///			m_pBuffer = p; m_extBuffer = true;
///			m_allocSize = rows * cols;
///		}
///		else { size_t n = Create(rows, cols); if (n) memcpy(m_pBuffer, p, n*sizeof(ElemType)); }
///	}
///	else if (m_format & matrixFormatBlock)
///	{
///		// sparse block
///		NOT_IMPLEMENTED
///	}
///	else
///	{
///		// sparse compressed
///		NOT_IMPLEMENTED
///	}
///}

///template<class ElemType>
///void BaseMatrixStorage<ElemType>::Assign(const BaseMatrixStorage<ElemType>& bms)
///{
///	if (&bms == this) return;
///
///	Init(bms.m_format, bms.m_deviceId);
///	size_t n = bms.m_numRows * bms.m_numCols;
///	if (n == 0) { m_numRows = bms.m_numRows; m_numCols = bms.m_numCols; return; }
///
///	if (bms.IsDenseFormat())
///	{
///		m_numRows = bms.m_numRows; m_numCols = bms.m_numCols;
///		if (bms.m_extBuffer) { m_pBuffer = bms.m_pBuffer; m_extBuffer = true; }
///		else memcpy(m_pBuffer=new ElemType[m_allocSize=n], bms.m_pBuffer, n*sizeof(ElemType));
///	}
///	else if (bms.IsSparseFormat())
///	{
///		Create(bms.m_numRows, bms.m_numCols);
///		size_t nc = (m_format & matrixFormatRowMajor) ? m_numRows : m_numCols;
///		size_t nr = (m_format & matrixFormatRowMajor) ? m_numCols : m_numRows;
///
///		memcpy(m_compPos, bms.m_compPos, (nc+1)*sizeof(index_t));
///		size_t n = m_compPos[nc]; Allocate(max(n,nr)); if (n==0) return;
///		memcpy(m_pBuffer, bms.m_pBuffer, n*sizeof(ElemType));
///		memcpy(m_compId, bms.m_compId, n*sizeof(index_t));
///	}
///	else if (bms.m_blockId)
///	{
///		// block (old)
///		Create(bms.m_numRows, bms.m_numCols);
///		size_t nc = (m_format & matrixFormatRowMajor) ? m_numRows : m_numCols;
///		size_t nr = (m_format & matrixFormatRowMajor) ? m_numCols : m_numRows;
///
///		size_t n = bms.m_blockCnt; Allocate(n==0 ? nr : n*nr);
///		for (size_t j=0; j<bms.m_blockCnt; ++j)
///		{
///			size_t i = bms.m_blockId[j];// - bms.m_blockShift;
///			memcpy(m_pBuffer+m_blockCnt*nr, bms.m_pBuffer+j*nr, nr*sizeof(ElemType));
///			m_blockPos[i] = m_blockCnt++;
///		}
///	}
///	else
///	{
///		// block (new)
///		Create(bms.m_numRows, bms.m_numCols); if (bms.m_blockCnt==0) return;
///		size_t nc = (m_format & matrixFormatRowMajor) ? m_numRows : m_numCols;
///		size_t nr = (m_format & matrixFormatRowMajor) ? m_numCols : m_numRows;
///
///		size_t n = bms.m_blockCnt*nr; Allocate(n);
///		memcpy(m_pBuffer, bms.m_pBuffer, n*sizeof(ElemType));
///		memcpy(m_blockPos, bms.m_blockPos, m_compSize*sizeof(size_t));
///		m_blockCnt = bms.m_blockCnt;
///	}
///}
///
///template<class ElemType>
///void BaseMatrixStorage<ElemType>::MakeDenseFrom(const BaseMatrixStorage<ElemType>& bms)
///{
///	if (&bms == this) return;
///
///	bool rmf = bms.IsRowMajor();
///	Init(rmf ? matrixFormatDenseRow : matrixFormatDenseCol);
///	size_t n = Create(bms.m_numRows,bms.m_numCols); if (n==0) return;
///	size_t nc = (rmf) ? bms.m_numRows : bms.m_numCols;
///	size_t nr = (rmf) ? bms.m_numCols : bms.m_numRows;
///
///	if (bms.IsDenseFormat()) memcpy(m_pBuffer, bms.m_pBuffer, n*sizeof(ElemType));
///	else if (bms.IsSparseFormat())
///	{
///		for (size_t j=0; j<nc; ++j)
///		{
///			size_t ns = bms.m_compPos[j], ne = bms.m_compPos[j+1];
///			for (size_t k=ns; k<ne; ++k)
///				m_pBuffer[j*nr+bms.m_compId[k]] = bms.m_pBuffer[k];
///		}
///	}
///	else if (bms.m_blockPos)
///	{
///		for (size_t j=0; j<nc; ++j)
///		{
///			size_t k = bms.m_blockPos[j]; if (k==string::npos) continue;
///			memcpy(m_pBuffer+j*nr, bms.m_pBuffer+k*nr, nr*sizeof(ElemType));
///		}
///	}
///	else if (bms.m_blockId)
///	{
///		for (size_t j=0; j<bms.m_blockCnt; ++j)
///		{
///			size_t i = bms.m_blockId[j];
///			memcpy(m_pBuffer+i*nr, bms.m_pBuffer+j*nr, nr*sizeof(ElemType));
///		}
///	}
///}
///
///template<class ElemType>
///void BaseMatrixStorage<ElemType>::MakeSparseFrom(const BaseMatrixStorage<ElemType>& bms)
///{
///	if (&bms == this) return;
///
///	bool rmf = bms.IsRowMajor();
///	Init(rmf ? matrixFormatSparseCSR : matrixFormatSparseCSC);
///	size_t n = Create(bms.m_numRows,bms.m_numCols); if (n==0) return;
///	size_t nc = (rmf) ? bms.m_numRows : bms.m_numCols;
///	size_t nr = (rmf) ? bms.m_numCols : bms.m_numRows;
///
///	if (bms.IsDenseFormat())
///	{
///		size_t m = max(nc,nr);
///		n = size_t(0.1*nc*nr + 0.5);
///		Allocate(max(n,m)); n = 0;
///		const ElemType* pi = bms.m_pBuffer;
///		for (size_t j=0; j<nc; ++j)
///		{
///			m_compPos[j] = (index_t)n;
///			for (size_t i=0; i<nr; ++i)
///			{
///				if (*pi == 0) { ++pi; continue; }
///				if (n==m_allocSize) Allocate(m_allocSize+max(nc,nr));
///				m_pBuffer[n] = *pi++; m_compId[n++] = (index_t)i;
///			}
///		}
///		m_compPos[nc] = (index_t)n;
///	}
///	else if (bms.IsSparseFormat())
///	{
///		n = bms.m_compPos[nc]; if (n==0) return;
///
///		Allocate(n);
///		memcpy(m_compPos, bms.m_compPos, (nc+1)*sizeof(index_t));
///		memcpy(m_compId, bms.m_compId, n*sizeof(index_t));
///		memcpy(m_pBuffer, bms.m_pBuffer, n*sizeof(ElemType));
///	}
///	else if (bms.m_blockPos)
///	{
///		if (bms.m_blockCnt==0) return;
///
///		size_t m = max(nc,nr);
///		n = size_t(0.1*bms.m_blockCnt*nr + 0.5);
///		Allocate(max(n,m)); n = 0;
///		const ElemType* pi = bms.m_pBuffer;
///		for (size_t j=0; j<nc; ++j)
///		{
///			m_compPos[j] = (index_t)n;
///			size_t k = bms.m_blockPos[j]; if (k==string::npos) continue;
///			for (size_t i=0; i<nr; ++i)
///			{
///				if (*pi==0) { ++pi; continue; }
///				if (n==m_allocSize) Allocate(m_allocSize+max(nc,nr));
///				m_pBuffer[n] = *pi++; m_compId[n++] = (index_t)i;
///			}
///		}
///		m_compPos[nc] = (index_t)n;
///	}
///}
///
///template<class ElemType>
///void BaseMatrixStorage<ElemType>::MakeBlockFrom(const BaseMatrixStorage<ElemType>& bms)
///{
///	if (&bms == this) return;
///
///	bool rmf = bms.IsRowMajor();
///	Init(rmf ? matrixFormatSparseBSR : matrixFormatSparseBSC);
///	size_t n = Create(bms.m_numRows,bms.m_numCols); if (n==0) return;
///	size_t nc = (rmf) ? bms.m_numRows : bms.m_numCols;
///	size_t nr = (rmf) ? bms.m_numCols : bms.m_numRows;
///
///	if (bms.IsDenseFormat())
///	{
///		vector<size_t> blk; blk.reserve(nc);
///		for (size_t j=0; j<nc; ++j)
///		{
///			const ElemType* pi = bms.m_pBuffer + j*nr;
///			for (size_t i=0; i<nr; ++i) if (*pi==0) ++pi; else { blk.push_back(j); break; }
///		}
///		if (blk.size()==0) return;
///
///		Allocate(blk.size()*nr);
///		for (size_t j=0; j<blk.size(); ++j)
///		{
///			size_t k = blk[j]; m_blockPos[k] = j;
///			memcpy(m_pBuffer+j*nr, bms.m_pBuffer+k*nr, nr*sizeof(ElemType));
///		}
///		m_blockCnt = blk.size();
///	}
///	else if (bms.IsSparseFormat())
///	{
///		n = 0;
///		for (size_t j=0; j<nc; ++j)
///			if (bms.m_compPos[j]<bms.m_compPos[j+1]) ++n;
///		if (n==0) return;
///
///		Allocate(n*nr);
///		ElemType* po = m_pBuffer;
///		for (size_t j=0; j<nc; ++j)
///		{
///			size_t ns = bms.m_compPos[j], ne = bms.m_compPos[j+1]; if (ns==ne) continue;
///			for (size_t k=ns; k<ne; ++k) po[bms.m_compId[k]] = bms.m_pBuffer[k];
///			m_blockPos[j] = m_blockCnt++; po += nr;
///		}
///	}
///	else if (bms.m_blockPos)
///	{
///		if (bms.m_blockCnt==0) return;
///
///		Allocate(bms.m_blockCnt*nr);
///		memcpy(m_blockPos, bms.m_blockPos, nc*sizeof(size_t));
///		memcpy(m_pBuffer, bms.m_pBuffer, bms.m_blockCnt*nr*sizeof(ElemType));
///		m_blockCnt = bms.m_blockCnt;
///	}
///}
///
///template<class ElemType>
///void BaseMatrixStorage<ElemType>::MakeFullBlock()
///{
///	bool rmf = IsRowMajor();
///	size_t nc = (rmf) ? m_numRows : m_numCols;
///	size_t nr = (rmf) ? m_numCols : m_numRows;
///	if (IsDenseFormat())
///	{
///		m_format = (rmf) ? matrixFormatSparseBSR : matrixFormatSparseBSC;
///		m_blockPos = new size_t[m_compSize=max(nc,nr)];
///		for (size_t j=0; j<nc; ++j) m_blockPos[j] = j;
///		m_blockCnt = nc;
///	}
///	else if (IsSparseFormat())
///	{
///		ElemType* pBuffer = new ElemType[m_allocSize=nc*nr];
///		memset(pBuffer, 0, m_allocSize*sizeof(ElemType));
///		m_blockPos = new size_t[m_compSize=max(nc,nr)];
///		for (size_t j=0; j<nc; ++j)
///		{
///			m_blockPos[j] = j;
///			size_t ns = m_compPos[j], ne = m_compPos[j+1]; if (ns==ne) continue;
///			ElemType* p = pBuffer + j*nr;
///			for (size_t k=ns; k<ne; ++k) p[m_compId[k]] = m_pBuffer[k];
///		}
///		m_format = (rmf) ? matrixFormatSparseBSR : matrixFormatSparseBSC;
///		delete[] m_pBuffer; m_pBuffer = pBuffer;
///		delete[] m_compPos; m_compPos = nullptr;
///		delete[] m_compId; m_compId = nullptr;
///		m_blockCnt = nc;
///	}
///	else
///	{
///		size_t n = 0; while (n<nc && m_blockPos[n]==n) ++n;
///		//for (size_t j=0; j<nc; ++j,++n) if (m_blockPos[j]!=j) break;
///		if (n==nc) return;
///
///		ElemType* pBuffer = new ElemType[m_allocSize=nc*nr];
///		for (size_t j=0; j<nc; ++j)
///		{
///			size_t k = m_blockPos[j]; m_blockPos[j] = j;
///			if (k==string::npos) memset(pBuffer+j*nr, 0, nr*sizeof(ElemType));
///			else memcpy(pBuffer+j*nr, m_pBuffer+k*nr, nr*sizeof(ElemType));
///		}
///		delete[] m_pBuffer; m_pBuffer = pBuffer;
///		m_blockCnt = nc;
///	}
///}
///
///template<class ElemType>
///size_t BaseMatrixStorage<ElemType>::GetItemCount() const
///{
///	if (IsDenseFormat()) return m_numRows*m_numCols;
///	if (IsSparseFormat()) return m_compPos[IsRowMajor() ? m_numRows : m_numCols];
///	return m_blockCnt*(IsRowMajor() ? m_numCols : m_numRows);
///}
///
///template<class ElemType>
///void BaseMatrixStorage<ElemType>::GetBlockId(vector<size_t>& v) const
///{
///	v.clear();
///	size_t nc = (m_format & matrixFormatRowMajor) ? m_numRows : m_numCols;
///	if (m_blockId) { v.resize(nc, string::npos); for (size_t j=0; j<m_blockCnt; ++j) v[m_blockId[j]] = j; }
///	else if (m_blockPos) for (size_t j=0; j<nc; ++j) v.push_back(m_blockPos[j]);
///}
///
///template<class ElemType>
///void BaseMatrixStorage<ElemType>::TransposeFrom(const BaseMatrixStorage<ElemType>& bms, bool hdr)
///{
///	Init(bms.m_format);
///	Create(bms.m_numCols, bms.m_numRows);
///
///	bool rmf = IsRowMajor();
///	size_t nc = (m_format & matrixFormatRowMajor) ? bms.m_numRows : bms.m_numCols;
///	size_t nr = (m_format & matrixFormatRowMajor) ? bms.m_numCols : bms.m_numRows;
///
///	if (IsDenseFormat())
///	{
///		const ElemType* src = bms.m_pBuffer;
///		for (size_t j=0; j<nc; ++j)
///		{
///			ElemType* po = m_pBuffer + j;
///			for (size_t i=0; i<nr; ++i) { *po = *src++; po += nc; }
///		}
///	}
///	else if (IsSparseFormat() && bms.m_compPos[nc]>0)
///	{
///		Allocate(bms.m_compPos[nc]);
///		for (size_t j=0; j<nc; ++j)
///		{
///			size_t ns = bms.m_compPos[j], ne = bms.m_compPos[j+1];
///			if (rmf) for (size_t i=ns; i<ne; ++i) PutItem(bms.m_compId[i], j, bms.m_pBuffer[i]);
///			else for (size_t i=ns; i<ne; ++i) PutItem(j, bms.m_compId[i], bms.m_pBuffer[i]);
///		}
///	}
///	else if (IsBlockFormat() && bms.m_blockCnt>0)
///	{
///		Allocate(bms.m_blockCnt*nr);
///		for (size_t j=0; j<nc; ++j)
///		{
///			size_t k = bms.m_blockPos[j]; if (k==string::npos) continue;
///			const ElemType* p = bms.m_pBuffer + k*nr;
///			if (rmf) for (size_t i=0; i<nr; ++i) { if (*p++) PutItem(i,j,p[-1]); }
///			else for (size_t i=0; i<nr; ++i) { if (*p++) PutItem(j,i,p[-1]); }
///		}
///	}
///	if (hdr) Transpose();
///}
///
///template<class ElemType>
///int BaseMatrixStorage<ElemType>::Compare(const BaseMatrixStorage<ElemType>& bms) const
///{
///	if (bms.m_format!=m_format) return diffFormat;
///	if (bms.m_deviceId!=m_deviceId) return diffDevice;
///	if (bms.m_numRows!=m_numRows) return diffRows;
///	if (bms.m_numCols!=m_numCols) return diffCols;
///
///	size_t n = m_numRows * m_numCols; if (n==0) return 0;
///	if (m_format & matrixFormatSparse)
///	{
///		size_t nc = (m_format & matrixFormatRowMajor) ? m_numRows : m_numCols;
///		size_t nr = (m_format & matrixFormatRowMajor) ? m_numCols : m_numRows;
///
///		if (IsSparseFormat())
///		{
///			int m = (m_compPos ? 1:0) + (bms.m_compPos ? 2:0); if (m==0) return 0;
///			if (m==1 || m==2 || memcmp(m_compPos, bms.m_compPos, (nc+1)*sizeof(index_t))) return diffIndex;
///			n = m_compPos[nc]; if (n==0) return 0;
///			if (memcmp(m_compId, bms.m_compId, n*sizeof(index_t))) return diffIndex;
///		}
///		else
///		{
///			if (m_blockCnt!=bms.m_blockCnt) return diffIndex;
///			if (m_blockCnt==0) return 0;
///
///			vector<size_t> v1; GetBlockId(v1);
///			vector<size_t> v2; bms.GetBlockId(v2);
///			const ElemType* src1 = m_pBuffer;
///			const ElemType* src2 = bms.m_pBuffer;
///			for (size_t j=0; j<v1.size(); ++j)
///			{
///				int m = (v1[j]==string::npos ? 0:1) + (v2[j]==string::npos ? 0:2);
///				if (m==1 || m==2) return diffIndex;
///				if (m==0) continue;
///
///				const ElemType* p1 = src1 + v1[j]*nr;
///				const ElemType* p2 = src2 + v2[j]*nr;
///				if (memcmp(p1, p2, nr*sizeof(ElemType)))
///				{
///					for (size_t i=0; i<nr; ++i) cout << " " << float(p1[i]); cout << endl;
///					for (size_t i=0; i<nr; ++i) cout << " " << float(p2[i]); cout << endl;
///					return diffData;
///				}
///			}
///			return 0;
///		}
///	}
///	// data
///	if (memcmp(m_pBuffer, bms.m_pBuffer, n*sizeof(ElemType))) return diffData;
///	return 0;
///}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//			get / put value
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

///template <class ElemType>
///ElemType BaseMatrixStorage<ElemType>::GetItem(size_t row, size_t col, size_t offset) const
///{
///	size_t nr = m_numRows, nc = m_numCols;
///	if (m_format & matrixFormatRowMajor) { size_t i = row; row = col; col = i; nr = m_numCols; nc = m_numRows; }
///
///	// dense
///	if (IsDenseFormat())
///	{
///		col += offset/nr; if (col>=nc || row>=nr) return 0;
///		return m_pBuffer[col*nr + row];
///	}
///	// sparse compressed
///	col += offset; if (col>=nc || row>=nr) return 0;
///	if ((m_format & matrixFormatBlock)==0)
///	{
///		size_t pos = m_compPos[col];
///		for (size_t fin=m_compPos[col+1]; pos<fin; ++pos)
///			if (m_compId[pos]==row) return m_pBuffer[pos];
///		return 0;
///	}
///	// sparse block (new)
///	if (m_blockPos)
///	{
///		size_t i = m_blockPos[col];
///		return (i==string::npos) ? 0 : m_pBuffer[i*nr + row];
///	}
///	// sparse block (old)
///	if (m_blockId)
///	{
///		for (size_t i=0; i<m_blockCnt; ++i)
///			if (m_blockId[i]==col) return m_pBuffer[i*nr + row];
///	}
///	return 0;
///}
///
///template <class ElemType>
///void BaseMatrixStorage<ElemType>::PutItem(size_t row, size_t col, ElemType val, size_t offset)
///{
///	// row/col priority
///	size_t nr = m_numRows, nc = m_numCols;
///	if (m_format & matrixFormatRowMajor) { size_t i = row; row = col; col = i; nr = m_numCols; nc = m_numRows; }
///
///	if (IsDenseFormat())
///	{
///		// dense
///		col += offset/nr; if (col>=nc || row>=nr) return;
///		m_pBuffer[col*nr + row] = val;
///	}
///	else if ((m_format & matrixFormatBlock)==0)
///	{
///		// sparse compressed
///		col += offset; if (col>=nc || row>=nr) return;
///		size_t pos = m_compPos[col];
///		for (size_t fin=m_compPos[col+1]; pos<fin; ++pos)
///			if (m_compId[pos]==row) { m_pBuffer[pos] = val; return; }
///
///		size_t n = m_compPos[nc]; if (n==m_allocSize) Allocate(m_allocSize+max(nr,nc));
///		for (size_t i=n; i>pos; --i) { m_pBuffer[i] = m_pBuffer[i-1]; m_compId[i] = m_compId[i-1]; }
///		for (size_t i=col+1; i<=nc; ++i) ++m_compPos[i];
///		m_pBuffer[pos] = val; m_compId[pos] = index_t(row);
///	}
///	else if (m_blockPos)
///	{
///		// sparse block (new)
///		col += offset; if (col>=nc || row>=nr) return;
///		size_t i = m_blockPos[col];
///		if (i==string::npos)
///		{
///			if ((m_blockCnt+1)*nr > m_allocSize)
///			{
///				size_t k = int(0.33*nc + 0.5); if (k==0) ++k;
///				Allocate(m_allocSize + k*nr);
///			}
///			m_blockPos[col] = i = m_blockCnt++;
///		}
///		m_pBuffer[i*nr + row] = val;
///	}
///	else if (m_blockId)
///	{
///		// sparse block (old)
///		col += offset; if (col>=nc || row>=nr) return;
///		ElemType* p = nullptr;
///		for (size_t j=0; j<m_blockCnt; ++j)
///			if (m_blockId[j] == col) { p = m_pBuffer + j*nr; break; }
///
///		if (p==nullptr)
///		{
///			if ((m_blockCnt+1)*nr> m_allocSize)
///			{
///				size_t k = int(0.33*nc + 0.5); if (k==0) ++k;
///				Allocate(m_allocSize + k*nr);
///			}
///			p = (ElemType*)memset(m_pBuffer+m_blockCnt*nr, 0, nr*sizeof(ElemType));
///			m_blockId[m_blockCnt++] = (index_t)col;
///		}
///		p[row] = val;
///	}
///}

template <class ElemType>
void BaseMatrixStorage<ElemType>::SetZeros()
{
	size_t n = m_numRows * m_numCols; if (n==0) return;
	if (IsDenseFormat()) memset(m_pBuffer, 0, n*sizeof(ElemType));
	else if (m_allocSize) memset(m_pBuffer, 0, m_allocSize*sizeof(ElemType));
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//			view
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

template<class ElemType>
string BaseMatrixStorage<ElemType>::Format() const
{
	if (m_format & matrixFormatSparse)
		return string("Sparse") + (m_format & matrixFormatBlock ? "BS":"CS") + (m_format & matrixFormatRowMajor ? "R":"C");
	return string("Dense") + (m_format & matrixFormatRowMajor ? "Row":"Col");
}

template <class ElemType>
string BaseMatrixStorage<ElemType>::GetInfo(bool all) const
{
	// data type
	string s;
	switch (sizeof(ElemType)) {
	case 2: s = "half"; break;
	case 4: s = "float"; break;
	case 8: s = "double"; break;
	default: s = "...";
	}
	// size + format
	char sz[32]; sprintf_s(sz, sizeof(sz), "[%d,%d] ", int(m_numRows), int(m_numCols));
	s += string(sz) + (m_format & matrixFormatBlock ? "b":"c")
					+ (m_format & matrixFormatSparse ? "s":"d")
					+ (m_format & matrixFormatRowMajor ? "r":"c");
	// device
	if (m_deviceId>=0) { sprintf_s(sz, sizeof(sz), "  GPU-%d", m_deviceId); s += sz; }
	else if (m_deviceId==CPUDEVICE) s += "  CPU";
	else if (m_deviceId==DEVICE_NOTSET) s += "  NotSet";
	else if (m_deviceId==DEVICE_AUTO) s += "  Auto";
	else { sprintf_s(sz, sizeof(sz), "  dev%d", m_deviceId); s += sz; }
	s += (m_format & matrixFormatRowMajor) ? "  Rows/Cols\n" : "  Cols/Rows\n";

	// data
	if (all)
	{
		size_t nc = (m_format & matrixFormatRowMajor) ? m_numRows : m_numCols;
		size_t nr = (m_format & matrixFormatRowMajor) ? m_numCols : m_numRows;

		int spb = m_format & matrixFormatSparseBlock;
		s += string("  buffer   = ") + hex(m_pBuffer) + "\t(" + long(m_allocSize) + ")";			// dense
		if (spb==matrixFormatSparseBlock) s += string("\tn=") + long(m_blockCnt*nr);				// block
		else if (spb==matrixFormatSparse) s += string("\tn=") + long(m_compPos ? m_compPos[nc]:0);	// sparse
		s += string("\n") +
				"  compPos  = " + hex(m_compPos) + "\t(" + long(m_compSize) + ")\tn=" + long(m_compPos ? nc+1:0) + "\n"
				"  compId   = " + hex(m_compId) + "\t(" + long(m_allocSize) + ")\tn=" + long(m_compPos ? m_compPos[nc]:0) + "\n"
				"  blockIds = " + hex(m_blockId) + "\t(" + long(m_compSize) + ")\tn=" + long(m_blockCnt) + "\n"
				"  blockPos = " + hex(m_blockPos) + "\t(" + long(m_compSize) + ")\tn=" + long(m_blockCnt) + "\n";
	}
	return s;
}

template <class ElemType>
void BaseMatrixStorage<ElemType>::ViewBuffer(ostream& os, const char* fmt, size_t pos, size_t len) const
{
	if (m_numRows*m_numCols == 0) return;

	if (IsDenseFormat()) ViewDense(os,fmt,pos,len);
	else if (IsSparseFormat()) ViewSparse(os,fmt,pos,len);
	else if (m_blockPos) ViewBlock2(os,fmt,pos,len);
	else ViewBlock1(os,fmt,pos,len);
}

template <class ElemType>
void BaseMatrixStorage<ElemType>::ViewDense(ostream& os, const char* fmt, size_t pos, size_t len) const
{
	size_t nc = (m_format & matrixFormatRowMajor) ? m_numRows : m_numCols;
	size_t nr = (m_format & matrixFormatRowMajor) ? m_numCols : m_numRows;
	pos /= nr; if (len > 0 && (pos + len) < nc) nc = pos + len;

	const ElemType* p = m_pBuffer + pos*nr;
	for (size_t j=pos; j<nc; ++j)
	{
		string s; char sz[32];
		for (size_t i=0; i<nr; ++i)
			{ sprintf_s(sz, sizeof(sz), fmt, float(*p++)); s += sz; }
		os << s << endl;
	}
}

template <class ElemType>
struct ElemItem
{
	size_t		row;
	size_t		col;
	ElemType	value;

	ElemItem() {}
	ElemItem(size_t r, ElemType v) : row(r), col(0), value(v) {}
	ElemItem(size_t r, size_t c, ElemType v) : row(r), col(c), value(v) {}

	bool operator < (const ElemItem& it) const { return row < it.row; }
};

template <class ElemType>
void BaseMatrixStorage<ElemType>::ViewSparse(ostream& os, const char* fmt, size_t pos, size_t len) const
{
	size_t nc = (m_format & matrixFormatRowMajor) ? m_numRows : m_numCols;
	size_t nr = (m_format & matrixFormatRowMajor) ? m_numCols : m_numRows;
	if (len > 0 && (pos + len) < nc) nc = pos + len;

	list<ElemItem<ElemType>> l;
	for (size_t j=pos; j<nc; ++j)
	{
		size_t k = m_compPos[j], fin = m_compPos[j+1];
		for (l.clear(); k<fin; ++k) l.push_back(ElemItem<ElemType>(m_compId[k],m_pBuffer[k]));
		if (l.size()>1) l.sort();

		string s; char sz[32];
		list<ElemItem<ElemType>>::iterator li = l.begin();
		for (size_t i=0; i<nr; ++i)
		{
			if (li==l.end()) sprintf_s(sz, sizeof(sz), fmt, 0.0);
			else if (i==(*li).row) sprintf_s(sz, sizeof(sz), fmt, float((*li++).value));
			else sprintf_s(sz, sizeof(sz), fmt, 0.0);
			s += sz;
		}
		os << s << endl;
	}
}

template <class ElemType>
void BaseMatrixStorage<ElemType>::ViewBlock1(ostream& os, const char* fmt, size_t pos, size_t len) const
{
	size_t nc = (m_format & matrixFormatRowMajor) ? m_numRows : m_numCols;
	size_t nr = (m_format & matrixFormatRowMajor) ? m_numCols : m_numRows;
	if (len > 0 && (pos + len) < nc) nc = pos + len;

	for (size_t j=pos; j<nc; ++j)
	{
		const ElemType* p = nullptr;
		for (size_t i=0; i<m_blockCnt; ++i)
			//if (m_blockId[i]-m_blockShift == j) { p = m_pBuffer + i*nr; break; }
			if (m_blockId[i] == j) { p = m_pBuffer + i*nr; break; }

		string s; char sz[32];
		if (p==nullptr) for (size_t i=0; i<nr; ++i) { sprintf_s(sz, sizeof(sz), fmt, 0.0); s += sz; }
		else for (size_t i=0; i<nr; ++i) { sprintf_s(sz, sizeof(sz), fmt, float(*p++)); s += sz; }
		os << s << endl;
	}
}

template <class ElemType>
void BaseMatrixStorage<ElemType>::ViewBlock2(ostream& os, const char* fmt, size_t pos, size_t len) const
{
	size_t nc = (m_format & matrixFormatRowMajor) ? m_numRows : m_numCols;
	size_t nr = (m_format & matrixFormatRowMajor) ? m_numCols : m_numRows;
	if (len > 0 && (pos + len) < nc) nc = pos + len;

	for (size_t j=pos; j<nc; ++j)
	{
		string s; char sz[32];
		if (m_blockPos[j]!=string::npos)
		{
			const ElemType* p = m_pBuffer + m_blockPos[j]*nr;
			for (size_t i=0; i<nr; ++i) { sprintf_s(sz, sizeof(sz), fmt, float(*p++)); s += sz; }
		}
		else for (size_t i=0; i<nr; ++i) { sprintf_s(sz, sizeof(sz), fmt, 0.0); s += sz; }
		os << s << endl;
	}
}

template <class ElemType>
void BaseMatrixStorage<ElemType>::ViewIds(ostream& os) const
{
	// dense
	if (IsDenseFormat()) return;

	// sparse block
	size_t nc = (m_format & matrixFormatRowMajor) ? m_numRows : m_numCols;
	if (m_format & matrixFormatBlock)
	{
		os << "  blockIds {";
		if (m_blockId)
		{
			for (size_t j=0; j<m_blockCnt; ++j) os << " " << m_blockId[j];
			//os << " }  " << m_blockCnt << "  shift=" << m_blockShift << endl;
			os << " }  " << m_blockCnt <<  endl;
		}
		else os << "}" << endl;

		os << "  blockPos {";
		if (m_blockPos)
		{
			for (size_t j=0; j<nc; ++j)
				if (m_blockPos[j]==string::npos) os << " -";
				else os << " " << m_blockPos[j];
			os << " }  " << m_blockCnt << endl;
		}
		else os << "}" << endl;
	}
	// sparse compressed
	else if (m_compPos)
	{
		size_t k = m_compPos[nc];
		os << "  compPos {"; for (size_t j=0; j<=nc; ++j) os << " " << m_compPos[j]; os << " }" << endl;
		os << "  compId  {"; for (size_t j=0; j<k; ++j) os << " " << m_compId[j]; os << " }" << endl;
	}
	else os << "  compPos {}" << endl
			<< "  compId  {}" << endl;
}

//==============================================================================
//		BaseMatrix
//		base class for all matrix types (CPU, GPU) x (dense, sparse)
//==============================================================================

template <class ElemType>
class MATH_API BaseMatrix
{
protected:
	size_t			m_numRows;
	size_t			m_numCols;
	size_t			m_sliceOffset;

	shared_ptr<BaseMatrixStorage<ElemType>> m_sob;

public:
	BaseMatrix() { Init(matrixFormatDense, CPUDEVICE); }
	BaseMatrix(MatrixFormat mft, device_t dev=CPUDEVICE) { Init(mft, dev); }
	virtual ~BaseMatrix() { Release(); }

	void Release() { m_numRows = m_numCols = m_sliceOffset = 0; m_sob = nullptr; }
	void Allocate(size_t n) { m_sob->Allocate(n); }
	void Init(MatrixFormat mft, device_t dev=CPUDEVICE);
	void Init() { m_sob->SetZeros(); }
	void Reset() { m_sliceOffset = 0; m_sob->Reset(); }

	//void Assign(size_t rows, size_t cols, ElemType* p, int flags);
	//void Assign(const BaseMatrix<ElemType>& mat, bool shallow = false);
	//void Resize(size_t rows, size_t cols);
	//void ResizeBack() { m_numRows = m_sob->GetNumRows(); m_numCols = m_sob->GetNumCols(); m_sliceOffset = 0; }

	//string Format() const { return m_sob->Format(); }
	//MatrixFormat GetFormat() const { return m_sob->GetFormat(); }
	//device_t GetDeviceId() const { return m_sob->GetDeviceId(); }

	//size_t GetNumRows() const { return m_numRows; }
	//size_t GetNumCols() const { return m_numCols; }
	//size_t GetNumStorageRows() const { return m_sob->GetNumRows(); }
	//size_t GetNumStorageCols() const { return m_sob->GetNumCols(); }
	//size_t GetNumElements() const { return m_numRows * m_numCols; }
	//size_t GetSizeAllocated() const { return m_sob->GetSizeAllocated(); }
	//size_t GetTotalBufferSize() const { return m_numRows*m_numCols*sizeof(ElemType); }
	//size_t GetDiagSize() const { return m_numRows < m_numCols ? m_numRows : m_numCols; }
	//size_t NzCount() const;

	//bool IsColMajor() const { return (m_sob->GetFormat() & matrixFormatRowMajor)==0; }
	//bool IsRowMajor() const { return (m_sob->GetFormat() & matrixFormatRowMajor)!=0; }
	//bool IsEmpty() const { return m_numRows == 0 || m_numCols == 0; }
	//bool HasExternalBuffer() const { return m_sob->HasExternalBuffer(); }
	//bool OwnBuffer() const { return !m_sob->HasExternalBuffer(); }
	//bool IsView() const { return (m_numRows != m_sob->GetNumRows() || m_numCols != m_sob->GetNumCols() || m_sliceOffset != 0); }

	//ElemType* Buffer() const { return m_sob->GetBuffer(); }						// dense (old)
	//ElemType* Data() const { return m_sob->GetBuffer() + m_sliceOffset; }		// dense (old)

	//ElemType* GetBuffer() const { return m_sob->GetBuffer(); }					// dense
	//ElemType* GetData() const { return m_sob->GetBuffer() + m_sliceOffset; }	// dense

	//ElemType GetItem(size_t row, size_t col) const { return m_sob->GetItem(row, col, m_sliceOffset); }
	//void PutItem(size_t row, size_t col, ElemType val) { m_sob->PutItem(row, col, val, m_sliceOffset); }

	//void TransposeFrom(const BaseMatrix<ElemType>& mat, bool hdr=false) { m_sob->TransposeFrom(*mat.m_sob.get(),hdr); ResizeBack(); }

	//void CopyToDense(BaseMatrix<ElemType>& mat) const;
	//void CopyToSparse(BaseMatrix<ElemType>& mat) const;
	//void CopyToBlock(BaseMatrix<ElemType>& mat) const;

	//int  Compare(const BaseMatrix<ElemType>& mat) const;

	// sparse format
	//int    GetColIdx() const { return m_sob->GetColIdx(); }
	//size_t GetCompPosSize() const { return m_sob->GetCompPosSize(); }
	//size_t GetBlockCount() const { return m_sob->GetBlockCount(); }
	//size_t GetBlockIdShift() const { return m_sob->GetBlockIdShift(); }

	//index_t* GetPrimePos() const { return m_sob->GetCompPos() + m_sliceOffset; }
	//index_t* GetCompPos() const { return m_sob->GetCompPos(); }
	//index_t* GetCompId() const { return m_sob->GetCompId(); }

	//size_t* GetBlockId() const { return m_sob->GetBlockId(); }
	//size_t* GetBlockPos() const { return m_sob->GetBlockPos() + m_sliceOffset; }
	//ElemType* GetNzValues() { return m_sob->GetNzValues(); }

	//void SetFormat(MatrixFormat mft) { m_sob->SetFormat(mft); }
	//void SetDeviceId(device_t dev) { m_sob->SetDeviceId(dev); }

	//void SetNumRows(size_t numRows) { m_numRows = numRows; }
	//void SetNumCols(size_t numCols) { m_numCols = numCols; }
	//void SetNumStorageRows(size_t rows) { m_sob->SetNumRows(rows); }
	//void SetNumStorageCols(size_t cols) { m_sob->SetNumCols(cols); }
	//void SetSizeAllocated(size_t alloc) { m_sob->SetSizeAllocated(alloc); }

	//void SetSlice(size_t start, size_t len);
	//void SetColumnSlice(size_t start, size_t len);

	//void SetBuffer(ElemType* parray, size_t alloc, bool external = false) { m_sob->SetBuffer(parray, alloc, external); }

	// sparse format
	//void SetColIdx(int idx) { m_sob->SetColIdx(idx); }
	//void SetCompPosSize(size_t indexSize) { m_sob->SetCompPosSize(indexSize); }
	//void SetBlockCount(size_t blockSize) { m_sob->SetBlockCount(blockSize); }

	//void SetCompPos(index_t* parray) { m_sob->SetCompPos(parray); }
	//void SetCompId(index_t* parray) { m_sob->SetCompId(parray); }
	//void SetBlockId(size_t* blockIds) { m_sob->SetBlockId(blockIds); }

	// gpu
	//void* GetTempHostBuffer() const { return m_sob->GetTempHostBuffer(); }
	//size_t GetTempHostBufferSize() const { return m_sob->GetTempHostBufferSize(); }
	//index_t* GetTempDeviceBuffer() const { return m_sob->GetTempDeviceBuffer(); }

	//void SetTempHostBuffer(void* buffer) { m_sob->SetTempHostBuffer(buffer); };
	//void SetTempHostBufferSize(size_t bufferSize) { m_sob->SetTempHostBufferSize(bufferSize); }
	//void ReserveTempDeviceBuffer(size_t minSize) const { m_sob->ReserveTempDeviceBuffer(minSize); }

	// others
	//void ShallowCopyFrom(const BaseMatrix& other) { *this = other; }
	//void ReleaseStorageMemory() { m_sob->Release(); }

	// view
	string GetInfo(bool all=true) const
	{
		char sz[64]; sprintf_s(sz, sizeof(sz), "[%lld,%lld]  offset=%lld  ", m_numRows, m_numCols, m_sliceOffset);
		return string(sz) + (m_sob==nullptr ? "***" : m_sob->GetInfo(all).c_str());
	}
	void ViewBuffer(ostream& os, const char* fmt="%6.2f") const { if (m_numRows*m_numCols > 0) m_sob->ViewBuffer(os, fmt); }
	void ViewData(ostream& os, const char* fmt="%6.2f") const;
	void ViewIds(ostream& os) const { m_sob->ViewIds(os); }

	// verification
	//void VerifyResizable(const char* function) const
	//{
	//	if (!m_sob.unique())
	//		LogicError("%s: Cannot resize the matrix because it is a view.", function);
	//	if (m_sob->HasExternalBuffer())
	//		LogicError("%s: Cannot resize the matrix because it is externally owned.", function);
	//}
	// same as VerifyResizable() except for the error message. Could be folded into one.
	//void VerifyMigratable(const char* function) const
	//{
	//	if (!m_sob.unique())
	//		LogicError("%s: Cannot migrate the matrix between devices because it is a view.", function);
	//	if (m_sob->HasExternalBuffer())
	//		LogicError("%s: Cannot migrate the matrix between devices because it is externally owned.", function);
	//}
	// This is needed for Sparse Matrices to ensure they can write to the matrix. Note: writing to slices is not currently supported
	//void VerifyWritable(const char* function) const
	//{
	//	if (m_numRows!=m_sob->GetNumRows() || m_numCols!=m_sob->GetNumCols())
	//		LogicError("%s: Cannot write to the matrix because it is a slice.", function);
	//}
	//void VerifySize(size_t rows, size_t cols)
	//{
	//	if (rows != m_numRows || cols != m_numCols)
	//		LogicError("VerifySize: expected matrix size %lu x %lu, but it is %lu x %lu", rows, cols, m_numRows, m_numCols);
	//}
};

} } }
