//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "ExportAPI.h"
#include "../Common/Basics.h"
#include "../Common/BaseTypes.h"
//#include <string>
//#include <stdint.h>
//#include <memory>
//#include <unordered_map>
//#include <map>
#include <list>

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

inline bool IsCPU(device_t dev) { return dev < 0; }
inline bool IsGPU(device_t dev) { return dev >= 0; }

inline size_t getbuffsize(size_t n, size_t m=2) { return (n + (m-1)) & ~(m-1); }

//==============================================================================
//		TracingGPUMemoryAllocator
//==============================================================================

//class MATH_API TracingGPUMemoryAllocator
//{
//private:
//	static int m_traceLevel;
//
//public:
//	static void SetTraceLevel(int traceLevel);
//	static bool IsTraceEnabled();
//
//	template <typename AllocatedElemType>
//	static AllocatedElemType* Allocate(device_t dev, size_t numRows, size_t numCols);
//
//	template <typename AllocatedElemType>
//	static AllocatedElemType* Allocate(device_t dev, size_t numElements);
//
//	template <typename AllocatedElemType>
//	static void Free(device_t dev, AllocatedElemType* bufferPtr, bool ignoreCUDARetCode = false);
//
//	// Let it be public method, the memory manager could check the totoal free memory and
//	// decide whether to physically release all the cached memory.
//	static std::pair<size_t, size_t> GetFreeAndTotalMemoryInMBs(device_t dev);
//
//private:
//	template <typename AllocatedElemType>
//	static AllocatedElemType* AllocateNoTrace(device_t dev, size_t numElements);
//};

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
    opLess, opEqual, opGreater, opGreaterEqual, opNotEqual, opLessEqual, // Note: must obey this order: (sgn(a-b) == -1, 0, +1), (sgn(a-b)!=-1, 0, +1)
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

enum class MatrixOrder
{
    RowMajor = 101,
    ColMajor = 102
};

enum class MatrixTranspose : char
{
    NoTrans = 'N',
    Trans = 'T',
    ConjTrans = 'C'
};

enum class SymMatrixType : char
{
    Up = 'U',				// symmetric matrix is stored in the upper part
    Low = 'L',				// symmetric matrix is stored in the lower part
    Full = 'F',				// full populated
    NotSymmetric = 'N'		// not a symmetric matrix
};

enum class MatrixOpSide : char
{
    Left = 'L',				// left multiply
    Right = 'R',			// right multiply
};

enum MatrixFormat : int
{
	matrixFormatDense			= 0,			// default is dense
	matrixFormatColMajor		= 0,			// default is column major
	matrixFormatRowMajor		= 0x0001,		// row major matrix
	matrixFormatSparse			= 0x0002,		// sparse matrix
	matrixFormatCompressed		= 0x0004,
	matrixFormatBlock			= 0x0008,

	matrixFormatDenseCol		= matrixFormatDense + matrixFormatColMajor,
	matrixFormatDenseRow		= matrixFormatDense + matrixFormatRowMajor,
	matrixFormatSparseCSC		= matrixFormatSparse + matrixFormatColMajor + matrixFormatCompressed,
	matrixFormatSparseCSR		= matrixFormatSparse + matrixFormatRowMajor + matrixFormatCompressed,
	matrixFormatSparseBSC		= matrixFormatSparse + matrixFormatColMajor + matrixFormatBlock,
	matrixFormatSparseBSR		= matrixFormatSparse + matrixFormatRowMajor + matrixFormatBlock,

	matrixFormatMask			= matrixFormatSparse + matrixFormatRowMajor + matrixFormatBlock,
	matrixFormatSparseBlock		= matrixFormatSparse + matrixFormatBlock,
};

enum MatrixFlags : int
{
	matrixFlagNone				= 0,
	matrixFlagRowMajor			= matrixFormatRowMajor,
	matrixFlagBlock				= matrixFormatBlock,
	matrixFlagBlockRow			= matrixFormatBlock + matrixFormatRowMajor,
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
//		sparse data support
//==============================================================================

template <class ElemType>
struct ElemItem
{
	index_t		row;
	index_t		col;
	ElemType	value;

	ElemItem() {}
	ElemItem(size_t r, ElemType v) : row((index_t)r), col(0), value(v) {}
	ElemItem(size_t r, size_t c, ElemType v) : row((index_t)r), col((index_t)c), value(v) {}

	bool operator < (const ElemItem& item) const { return row < item.row; }
	bool operator == (const ElemItem& item) const { return row==item.row && col==item.col && value==item.value; }
};

template <class ElemType>
class SparseData : public vector<ElemItem<ElemType>>
{
	typedef vector<ElemItem<ElemType>> Base;
	int		format;
	bool	sorted;

public:
	SparseData(int fmt=0) : format(fmt), sorted(false) {}

	int  GetFormat() const { return format; }
	bool IsSorted() const { return sorted; }
	bool IsRowMajor() const { return (format & matrixFormatRowMajor)!=0; }
	bool IsColMajor() const { return (format & matrixFormatRowMajor)==0; }

	void Init(int fmt) { format = fmt; sorted = false; clear(); }
	void push_back(const ElemItem<ElemType>& val) { Base::push_back(val); sorted = false; }
	void SetSorted() { sorted = true; }

	void SortByRows()
	{
		if (sorted && IsRowMajor()) return;
		if (size()>1) std::sort(begin(), end(), [](const ElemItem<ElemType>& a, const ElemItem<ElemType>& b)
													{ return (a.row==b.row) ? (a.col<b.col) : (a.row<b.row); });
		format |= matrixFormatRowMajor;
		sorted = true;
	}
	void SortByCols()
	{
		if (sorted && IsColMajor()) return;
		if (size()>1) std::sort(begin(), end(), [](const ElemItem<ElemType>& a, const ElemItem<ElemType>& b)
													{ return (a.col==b.col) ? (a.row<b.row) : (a.col<b.col); });
		format &= ~matrixFormatRowMajor;
		sorted = true;
	}
	void ViewIndex(ostream& os, int limit=16) const
	{
		int n = 0;
		for (const_iterator i=begin(); i!=end(); ++i)
		{
			os << " (" << (*i).row << "," << (*i).col << ")";
			if (++n==limit) { os << endl; n = 0; }
		}
		if (n) os << endl;
	}
	void View(ostream& os) const
	{
		for (const_iterator i=begin(); i!=end(); ++i)
			os << "\t" << (*i).row << "\t" << (*i).col << "\t" << float((*i).value) << endl;
	}
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
	size_t			m_buffSize;				// allocated size
	ElemType*		m_pBuffer;
	bool			m_extBuffer;

	// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
	//		CPUSparseMatrix variables
	// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

	index_t*		m_compPos;				// position in Sparse format
	index_t*		m_compId;				// row/col ids in CSC/CSR format
	size_t			m_compSize;				// allocated size
	size_t			m_blockCnt;

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
	BaseMatrixStorage(MatrixFormat mft=matrixFormatDense, device_t dev=CPUDEVICE) { ZeroInit(mft, dev); }
	~BaseMatrixStorage() { Release(); }

	void Release(bool all=true);
	void Allocate(size_t n);
	void Reset();

	void Init(MatrixFormat mft, device_t dev=CPUDEVICE);
	size_t Create(size_t rows, size_t cols);
	void Create(size_t rows, size_t cols, ElemType* p, int flags);
	void Assign(const BaseMatrixStorage<ElemType>& bms);

	void MakeDenseFrom(const BaseMatrixStorage<ElemType>& bms, size_t offset=0, size_t len=0);
	void MakeSparseFrom(const BaseMatrixStorage<ElemType>& bms, size_t offset=0, size_t len=0);
	void MakeBlockFrom(const BaseMatrixStorage<ElemType>& bms, size_t offset=0, size_t len=0);
	void MakeFullBlockFrom(const BaseMatrixStorage<ElemType>& bms, size_t offset=0, size_t len=0);

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
	//size_t GetSizeAllocated() const { return m_buffSize; }
	size_t GetItemCount() const;

	bool IsEmpty() const { return m_numRows == 0 || m_numCols == 0; }
	bool HasExternalBuffer() const { return m_extBuffer; }

	ElemType* GetBuffer() const { return m_pBuffer; }

	//void SetFormat(MatrixFormat mft) { m_format = mft; }
	//void SetDeviceId(device_t dev) { m_deviceId = dev; }

	//void SetNumRows(size_t rows) { m_numRows = rows; }
	//void SetNumCols(size_t cols) { m_numCols = cols; }
	//void SetSizeAllocated(size_t alloc) { m_buffSize = alloc; }

	//void SetBuffer(ElemType* p, size_t total, bool external = false) { m_pBuffer = p; /*m_totalSize = total;*/ m_extBuffer = external; }

	ElemType GetItem(size_t row, size_t col, size_t offset=0) const;
	void PutItem(size_t row, size_t col, ElemType val, size_t offset=0);

	ElemType& Item(size_t row, size_t col, size_t offset=0);

	void GetSparseData(SparseData<ElemType>& v, size_t offset=0, size_t len=0) const;
	void PutSparseData(const SparseData<ElemType>& spd);

	bool Reshape(size_t rows, size_t cols);
	void TransposeFrom(const BaseMatrixStorage<ElemType>& bms, bool hdr=false);
	void TransposeHeader() { size_t n = m_numRows; m_numRows = m_numCols; m_numCols = n; m_format = MatrixFormat(m_format ^ matrixFormatRowMajor); }

	int  Compare(const BaseMatrixStorage<ElemType>& bms) const;

	// CPU sparse format
	size_t GetBlockCount() const { return m_blockCnt; }
	//size_t GetCompPosSize() const { return m_compSize; }

	index_t* GetCompPos() const { return m_compPos; }
	index_t* GetCompId() const { return m_compId; }

	void SetBlockCount(size_t n) { m_blockCnt = n; }
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
	string FormatStr() const;
	string GetInfo(bool all=true) const;
	void ViewBuffer(ostream& os, const char* fmt = "%8.4f", size_t offset=0, size_t len=0) const;
	void ViewIds(ostream& os) const;

protected:
	void ZeroInit(MatrixFormat mft, device_t dev);
	void GetBlockId(vector<size_t>& v) const;
	void Transpose(size_t rows, size_t cols, const ElemType* p, int flags);

	void ViewDense(ostream& os, const char* fmt, size_t offset, size_t len) const;
	void ViewSparse(ostream& os, const char* fmt, size_t offset, size_t len) const;
	void ViewBlock(ostream& os, const char* fmt, size_t offset, size_t len) const;
};

template<class ElemType>
void BaseMatrixStorage<ElemType>::ZeroInit(MatrixFormat mft, device_t dev)
{
	m_format = mft;
	m_deviceId = dev;
	m_numRows = 0;
	m_numCols = 0;
	m_buffSize = 0;
	m_pBuffer = nullptr;
	m_extBuffer = false;

	// sparse matrix
	m_compSize = 0;
	m_compPos = nullptr;
	m_compId = nullptr;
	m_blockCnt = 0;

	// GPU support
	m_tempDeviceBuffer = nullptr;
	m_tempDeviceBufferSize = 0;
	m_tempHostBuffer = nullptr;
	m_tempHostBufferSize = 0;
}

template<class ElemType>
void BaseMatrixStorage<ElemType>::Release(bool all)
{
	if (IsCPU(m_deviceId))
	{
		if (m_extBuffer) { m_extBuffer = false; m_pBuffer = nullptr; m_buffSize = 0; }
		else if (all) { delete[] m_pBuffer; m_pBuffer = nullptr; m_buffSize = 0; }
		delete[] m_compPos; m_compPos = nullptr;
		delete[] m_compId; m_compId = nullptr;
		m_compSize = m_blockCnt = 0;
	}
	else
	{
#ifndef CPUONLY
		if (m_pBuffer!=nullptr)
			TracingGPUMemoryAllocator::Free<ElemType>(m_deviceId, m_pBuffer, true);
		m_pBuffer = nullptr;

		if (m_tempDeviceBuffer!=nullptr)
			TracingGPUMemoryAllocator::Free<index_t>(m_deviceId, m_tempDeviceBuffer, true);
		m_tempDeviceBuffer = nullptr;
		m_tempDeviceBufferSize = 0;
#endif
		delete[](byte*) m_tempHostBuffer; m_tempHostBuffer = nullptr;
	}
	m_numRows = m_numCols = 0;
}

template<class ElemType>
void BaseMatrixStorage<ElemType>::Allocate(size_t n)
{
	size_t k = getbuffsize(n);
	if (k>m_buffSize)
	{
		ElemType* pbuff = m_pBuffer; m_pBuffer = new ElemType[k];
		if (pbuff) memcpy(m_pBuffer, pbuff, m_buffSize*sizeof(ElemType));
		memset(m_pBuffer+m_buffSize, 0, (k-m_buffSize)*sizeof(ElemType));
		delete[] pbuff;

		if (IsSparseFormat())
		{
			index_t* prev = m_compId; m_compId = new index_t[k];
			if (prev) memcpy(m_compId, prev, m_buffSize*sizeof(index_t));
			delete[] prev;
		}
		m_buffSize = k;
	}
	else if (IsSparseFormat() && m_compId==nullptr) m_compId = new index_t[m_buffSize];
}

template<class ElemType>
void BaseMatrixStorage<ElemType>::Reset()
{
	if (IsDenseFormat()) { size_t n = m_numRows*m_numCols; if (n) memset(m_pBuffer, 0, n*sizeof(ElemType)); }
	else memset(m_compPos, (IsSparseFormat() ? 0 : 0xff), m_compSize*sizeof(index_t));
	m_blockCnt = 0;
}

template<class ElemType>
void BaseMatrixStorage<ElemType>::Init(MatrixFormat mft, device_t dev)
{
	m_format = mft; m_deviceId = dev;
	m_numRows = m_numCols = 0;
	if (m_extBuffer) { m_pBuffer = nullptr; m_extBuffer = false; m_buffSize = 0; }
	else if (m_buffSize) memset(m_pBuffer, 0, m_buffSize*sizeof(ElemType));

	if (IsDenseFormat())
	{
		delete[] m_compPos; m_compPos = nullptr;
		delete[] m_compId; m_compId = nullptr;
		m_compSize = m_blockCnt = 0;
	}
	else if (IsBlockFormat())
	{
		if (m_compSize) memset(m_compPos, 0xff, m_compSize*sizeof(index_t));
		delete[] m_compId; m_compId = nullptr;
		m_blockCnt = 0;
	}
	else if (m_compSize) memset(m_compPos, 0, m_compSize*sizeof(index_t));
}

template<class ElemType>
size_t BaseMatrixStorage<ElemType>::Create(size_t rows, size_t cols)
{
	size_t n = rows * cols;
	m_numRows = rows; m_numCols = cols;
	if (IsDenseFormat())
	{
		if (m_extBuffer) { m_pBuffer = nullptr; m_extBuffer = false; m_buffSize = 0; }
		if (n>m_buffSize) Allocate(n);
	}
	else if (IsSparseFormat())
	{
		size_t k = max(rows,cols) + 1;
		if (k>m_compSize) { delete[] m_compPos; m_compPos = new index_t[m_compSize=k]; }
		memset(m_compPos, 0, m_compSize*sizeof(index_t));
		if (m_buffSize && m_compId==nullptr) m_compId = new index_t[m_buffSize];
	}
	else if (IsBlockFormat())
	{
		size_t k = max(rows,cols) + 1;
		if (k>m_compSize) { delete[] m_compPos; m_compPos = new index_t[m_compSize=k]; }
		memset(m_compPos, 0xff, m_compSize*sizeof(index_t));
		m_blockCnt = 0;
	}
	return n;
}

template<class ElemType>
void BaseMatrixStorage<ElemType>::Create(size_t rows, size_t cols, ElemType* p, int flags)
{
	size_t n = Create(rows, cols); if (n==0) return;
	int k = (m_format & matrixFormatRowMajor ? 1:0) + (flags & matrixFormatRowMajor ? 1:0);
	if (k==1) { Transpose(rows,cols,p,flags); return; }

	//m_format = MatrixFormat(m_format | (flags & matrixFormatRowMajor));
	if (IsDenseFormat())
	{
		if (flags & matrixFlagExternalBuffer)
		{
			m_numRows = rows; m_numCols = cols;
			m_pBuffer = p; m_extBuffer = true;
			m_buffSize = rows * cols;
		}
		else if (n) memcpy(m_pBuffer, p, n*sizeof(ElemType));
		return;
	}
	bool rmf = IsRowMajor();
	size_t nc = (rmf) ? rows : cols;
	size_t nr = (rmf) ? cols : rows;
	size_t m = size_t(0.2*nc + 0.5);
	Allocate(m = (m==0) ? nr : m*nr);

	if (IsSparseFormat())
	{
		for (size_t j=0,n=0; j<nc; ++j)
		{
			for (size_t i=0; i<nr; ++i)
			{
				if (*p++==0) continue;
				if (n==m_buffSize) Allocate(m_buffSize+m);
				m_pBuffer[n] = p[-1]; m_compId[n++] = (index_t)i;
			}
			m_compPos[j+1] = (index_t)n;
		}
	}
	else
	{
		for (size_t j=0; j<nc; ++j)
			if (rmf) for (size_t i=0; i<nr; ++i) { if (*p++!=0) PutItem(j,i,p[-1]); }
			else for (size_t i=0; i<nr; ++i) if (*p++!=0) PutItem(i,j,p[-1]);
	}
}

template<class ElemType>
void BaseMatrixStorage<ElemType>::Transpose(size_t rows, size_t cols, const ElemType* p, int flags)
{
	bool rmf = (flags & matrixFormatRowMajor)!=0;
	size_t nc = (rmf) ? cols : rows;
	size_t nr = (rmf) ? rows : cols;

	if (IsDenseFormat())
	{
		ElemType* po = m_pBuffer;
		for (size_t i=0; i<nc; ++i)
		{
			const ElemType* pi = p + i;
			for (size_t j=0; j<nr; ++j) { *po++ = *pi; pi += nc; }
		}
		return;
	}
	size_t m = size_t(0.2*nc + 0.5);
	Allocate(m = (m==0) ? nr : m*nr);

	if (IsSparseFormat())
	{
		for (size_t j=0,n=0; j<nc; ++j)
		{
			const ElemType* pj = p + j;
			for (size_t i=0; i<nr; ++i,pj+=nc)
			{
				if (*pj==0) continue;
				if (n==m_buffSize) Allocate(m_buffSize+m);
				m_pBuffer[n] = *pj; m_compId[n++] = (index_t)i;
			}
			m_compPos[j+1] = (index_t)n;
		}
	}
	else
	{
		for (size_t j=0; j<nc; ++j)
		{
			size_t k = m_compPos[j];
			ElemType* po = (k==string::npos) ? nullptr : m_pBuffer + k*nr;
			const ElemType* pj = p + j;
			for (size_t i=0; i<nr; ++i,pj+=nc)
			{
				if (*pj==0) continue;
				if (po==nullptr)
				{
					size_t k = m_blockCnt*nr;
					if (k+nr > m_buffSize) Allocate(k + m);
					po = m_pBuffer + k; m_compPos[j] = (index_t)m_blockCnt++;
				}
				po[i] = *pj;
			}
		}
	}
}

template<class ElemType>
void BaseMatrixStorage<ElemType>::Assign(const BaseMatrixStorage<ElemType>& bms)
{
	if (&bms == this) return;

	Init(bms.m_format, bms.m_deviceId);
	size_t n = bms.m_numRows * bms.m_numCols;
	if (n == 0) { m_numRows = bms.m_numRows; m_numCols = bms.m_numCols; return; }

	if (bms.IsDenseFormat())
	{
		m_numRows = bms.m_numRows; m_numCols = bms.m_numCols;
		if (bms.m_extBuffer) { m_pBuffer = bms.m_pBuffer; m_extBuffer = true; }
		else memcpy(m_pBuffer=new ElemType[m_buffSize=n], bms.m_pBuffer, n*sizeof(ElemType));
	}
	else if (bms.IsSparseFormat())
	{
		Create(bms.m_numRows, bms.m_numCols);
		size_t nc = (m_format & matrixFormatRowMajor) ? m_numRows : m_numCols;
		size_t nr = (m_format & matrixFormatRowMajor) ? m_numCols : m_numRows;

		memcpy(m_compPos, bms.m_compPos, (nc+1)*sizeof(index_t));
		size_t n = m_compPos[nc]; Allocate(max(n,nr)); if (n==0) return;
		memcpy(m_pBuffer, bms.m_pBuffer, n*sizeof(ElemType));
		memcpy(m_compId, bms.m_compId, n*sizeof(index_t));
	}
	else
	{
		Create(bms.m_numRows, bms.m_numCols); if (bms.m_blockCnt==0) return;
		size_t nc = (m_format & matrixFormatRowMajor) ? m_numRows : m_numCols;
		size_t nr = (m_format & matrixFormatRowMajor) ? m_numCols : m_numRows;

		size_t n = bms.m_blockCnt*nr; Allocate(n);
		memcpy(m_pBuffer, bms.m_pBuffer, n*sizeof(ElemType));
		memcpy(m_compPos, bms.m_compPos, m_compSize*sizeof(index_t));
		m_blockCnt = bms.m_blockCnt;
	}
}

template<class ElemType>
bool BaseMatrixStorage<ElemType>::Reshape(size_t rows, size_t cols)
{
	if (rows*cols != m_numRows*m_numCols) return false;
	if (rows==m_numRows && cols==m_numCols) return true;

	bool rmf = IsRowMajor();
	size_t nr1, nr2, nc1, nc2;
	if (rmf) { nc1 = m_numRows, nr1 = m_numCols; nc2 = rows; nr2 = cols; }
	else { nc1 = m_numCols, nr1 = m_numRows; nc2 = cols; nr2 = rows; }
	m_numRows = rows;
	m_numCols = cols;

	if (IsSparseFormat())
	{
		index_t* compPos = m_compPos; m_compPos = new index_t[m_compSize=nc2+1];
		memset(m_compPos, 0, m_compSize*sizeof(index_t));
		if (compPos[nc1]==0) { delete[] compPos; return true; }

		ElemType* pBuff = m_pBuffer; m_pBuffer = new ElemType[m_buffSize];
		index_t* compId = m_compId; m_compId = new index_t[m_buffSize];
		for (size_t j=0; j<nc1; ++j)
		{
			size_t ns = compPos[j], ne = compPos[j+1];
			for (size_t k=ns; k<ne; ++k)
			{
				size_t n = j*nr1 + compId[k];
				if (rmf) PutItem(n/nr2, n%nr2, pBuff[k]);
				else PutItem(n%nr2, n/nr2, pBuff[k]);
			}
		}
		delete[] compPos;
		delete[] compId;
		delete[] pBuff;
	}
	else if (IsBlockFormat())
	{
		index_t* compPos = m_compPos; m_compPos = new index_t[m_compSize=nc2+1];
		memset(m_compPos, 0xff, m_compSize*sizeof(index_t));
		if (m_blockCnt==0) { delete[] compPos; return true; }

		m_blockCnt = 0;
		ElemType* pBuff = m_pBuffer; m_pBuffer = new ElemType[m_buffSize];
		for (size_t j=0; j<nc1; ++j)
		{
			size_t k = compPos[j]; if (k==string::npos) continue;
			ElemType* p = pBuff + k*nr1;
			for (size_t i=0; i<nr1; ++i)
			{
				if (*p==0) { ++p; continue; }
				size_t n = j*nr1 + i;
				if (rmf) PutItem(n/nr2, n%nr2, *p++);
				else PutItem(n%nr2, n/nr2, *p++);
			}
		}
		delete[] compPos;
		delete[] pBuff;
	}
	return true;
}

template<class ElemType>
void BaseMatrixStorage<ElemType>::MakeDenseFrom(const BaseMatrixStorage<ElemType>& bms, size_t offset, size_t len)
{
	if (&bms == this) return;

	bool rmf = bms.IsRowMajor();
	size_t nc = (rmf) ? bms.m_numRows : bms.m_numCols;
	size_t nr = (rmf) ? bms.m_numCols : bms.m_numRows;
	if (bms.IsDenseFormat()) offset /= nr;
	if (len>0) nc = len;

	Init(rmf ? matrixFormatDenseRow : matrixFormatDenseCol);
	size_t n = (rmf) ? Create(nc,nr) : Create(nr,nc); if (n==0) return;

	if (bms.IsDenseFormat()) memcpy(m_pBuffer, bms.m_pBuffer+offset*nr, n*sizeof(ElemType));
	else if (bms.IsSparseFormat())
	{
		const index_t* compPos = bms.m_compPos + offset;
		size_t n = compPos[nc] - compPos[0]; if (n==0) return;
		for (size_t j=0; j<nc; ++j)
		{
			size_t ns = compPos[j], ne = compPos[j+1];
			for (size_t k=ns; k<ne; ++k)
				m_pBuffer[j*nr+bms.m_compId[k]] = bms.m_pBuffer[k];
		}
	}
	else if (bms.m_blockCnt>0)
	{
		const index_t* compPos = bms.m_compPos + offset;
		for (size_t j=0; j<nc; ++j)
		{
			size_t k = compPos[j]; if (k==string::npos) continue;
			memcpy(m_pBuffer+j*nr, bms.m_pBuffer+k*nr, nr*sizeof(ElemType));
		}
	}
}

template<class ElemType>
void BaseMatrixStorage<ElemType>::MakeSparseFrom(const BaseMatrixStorage<ElemType>& bms, size_t offset, size_t len)
{
	if (&bms == this) return;

	bool rmf = bms.IsRowMajor();
	size_t nc = (rmf) ? bms.m_numRows : bms.m_numCols;
	size_t nr = (rmf) ? bms.m_numCols : bms.m_numRows;
	if (bms.IsDenseFormat()) offset /= nr;
	if (len>0) nc = len;

	Init(rmf ? matrixFormatSparseCSR : matrixFormatSparseCSC);
	size_t n = (rmf) ? Create(nc,nr) : Create(nr,nc); if (n==0) return;

	if (bms.IsDenseFormat())
	{
		n = 0;
		size_t m = size_t(0.3*nc + 0.5);
		Allocate(m = (m==0) ? nr : m*nr);
		const ElemType* pi = bms.m_pBuffer + offset*nr;
		for (size_t j=0; j<nc; ++j)
		{
			m_compPos[j] = (index_t)n;
			for (size_t i=0; i<nr; ++i)
			{
				if (*pi == 0) { ++pi; continue; }
				if (n==m_buffSize) Allocate(m_buffSize + m);
				m_pBuffer[n] = *pi++; m_compId[n++] = (index_t)i;
			}
		}
		m_compPos[nc] = (index_t)n;
	}
	else if (bms.IsSparseFormat())
	{
		const index_t* compPos = bms.m_compPos + offset;
		size_t n = compPos[nc] - compPos[0]; if (n==0) return;

		Allocate(n);
		for (size_t i=0; i<=nc; ++i) m_compPos[i] = compPos[i] - compPos[0];
		memcpy(m_pBuffer, bms.m_pBuffer+compPos[0], n*sizeof(ElemType));
		memcpy(m_compId, bms.m_compId+compPos[0], n*sizeof(index_t));
	}
	else if (bms.m_compPos)
	{
		if (bms.m_blockCnt==0) return;

		n = 0;
		size_t m = size_t(0.3*nc + 0.5);
		Allocate(m = (m==0) ? nr : m*nr);
		const index_t* compPos = bms.m_compPos + offset;
		for (size_t j=0; j<nc; ++j)
		{
			m_compPos[j] = (index_t)n;
			size_t k = compPos[j]; if (k==string::npos) continue;
			const ElemType* pi = bms.m_pBuffer + k*nr;
			for (size_t i=0; i<nr; ++i)
			{
				if (*pi==0) { ++pi; continue; }
				if (n==m_buffSize) Allocate(m_buffSize + m);
				m_pBuffer[n] = *pi++; m_compId[n++] = (index_t)i;
			}
		}
		m_compPos[nc] = (index_t)n;
	}
}

template<class ElemType>
void BaseMatrixStorage<ElemType>::MakeBlockFrom(const BaseMatrixStorage<ElemType>& bms, size_t offset, size_t len)
{
	if (&bms == this) return;

	bool rmf = bms.IsRowMajor();
	size_t nc = (rmf) ? bms.m_numRows : bms.m_numCols;
	size_t nr = (rmf) ? bms.m_numCols : bms.m_numRows;
	if (bms.IsDenseFormat()) offset /= nr;
	if (len>0) nc = len;

	Init(rmf ? matrixFormatSparseBSR : matrixFormatSparseBSC);
	size_t n = (rmf) ? Create(nc,nr) : Create(nr,nc); if (n==0) return;

	if (bms.IsDenseFormat())
	{
		const ElemType* pbuff = bms.m_pBuffer + offset*nr;
		vector<size_t> blk; blk.reserve(nc);
		for (size_t j=0; j<nc; ++j)
		{
			const ElemType* pi = pbuff + j*nr;
			for (size_t i=0; i<nr; ++i) if (*pi++) { blk.push_back(j); break; }
		}
		if (blk.size()==0) return;

		Allocate(blk.size()*nr);
		for (size_t j=0; j<blk.size(); ++j)
		{
			m_compPos[j] = (index_t)j;
			memcpy(m_pBuffer+j*nr, pbuff+blk[j]*nr, nr*sizeof(ElemType));
		}
		m_blockCnt = blk.size();
	}
	else if (bms.IsSparseFormat())
	{
		n = 0;
		const index_t* compPos = bms.m_compPos + offset;
		for (size_t j=0; j<nc; ++j) if (compPos[j]<compPos[j+1]) ++n;
		if (n==0) return;

		Allocate(n*nr);
		ElemType* po = m_pBuffer;
		for (size_t j=0; j<nc; ++j)
		{
			size_t ns = compPos[j], ne = compPos[j+1]; if (ns==ne) continue;
			for (size_t k=ns; k<ne; ++k) po[bms.m_compId[k]] = bms.m_pBuffer[k];
			m_compPos[j] = (index_t)m_blockCnt++; po += nr;
		}
	}
	else if (bms.m_compPos)
	{
		n = 0;
		const index_t* compPos = bms.m_compPos + offset;
		for (size_t j=0; j<nc; ++j) if (compPos[j]>=0) ++n;
		if (n==0) return;

		Allocate(n*nr);
		ElemType* po = m_pBuffer;
		for (size_t j=0; j<nc; ++j)
		{
			size_t k = compPos[j]; if (k==string::npos) continue;
			memcpy(po, bms.m_pBuffer+k*nr, nr*sizeof(ElemType));
			m_compPos[j] = (index_t)m_blockCnt++; po += nr;
		}
	}
}

template<class ElemType>
void BaseMatrixStorage<ElemType>::MakeFullBlockFrom(const BaseMatrixStorage<ElemType>& bms, size_t offset, size_t len)
{
	bool rmf = bms.IsRowMajor();
	size_t nc = (rmf) ? bms.m_numRows : bms.m_numCols;
	size_t nr = (rmf) ? bms.m_numCols : bms.m_numRows;
	if (bms.IsDenseFormat()) offset /= nr;
	if (len>0) nc = len;

	Init(rmf ? matrixFormatSparseBSR : matrixFormatSparseBSC);
	size_t n = (rmf) ? Create(nc,nr) : Create(nr,nc); if (n==0) return;

	Allocate(n);
	for (size_t j=0; j<nc; ++j) m_compPos[j] = (index_t)j;
	m_blockCnt = nc;

	if (bms.IsDenseFormat())
	{
		memcpy(m_pBuffer, bms.m_pBuffer+offset*nr, n*sizeof(ElemType));
	}
	else if (bms.IsSparseFormat())
	{
		const index_t* compPos = bms.m_compPos + offset;
		for (size_t j=0; j<nc; ++j)
		{
			m_compPos[j] = (index_t)j;
			size_t ns = compPos[j], ne = compPos[j+1]; if (ns==ne) continue;
			ElemType* p = m_pBuffer + j*nr;
			for (size_t k=ns; k<ne; ++k) p[bms.m_compId[k]] = bms.m_pBuffer[k];
		}
	}
	else
	{
		const index_t* compPos = bms.m_compPos + offset;
		for (size_t j=0; j<nc; ++j)
		{
			m_compPos[j] = (index_t)j;
			size_t k = compPos[j]; if (k==string::npos) continue;
			memcpy(m_pBuffer+j*nr, bms.m_pBuffer+k*nr, nr*sizeof(ElemType));
		}
	}
}

template<class ElemType>
size_t BaseMatrixStorage<ElemType>::GetItemCount() const
{
	if (IsDenseFormat()) return m_numRows*m_numCols;
	if (IsSparseFormat()) return m_compPos[IsRowMajor() ? m_numRows : m_numCols];
	return m_blockCnt*(IsRowMajor() ? m_numCols : m_numRows);
}

template<class ElemType>
void BaseMatrixStorage<ElemType>::GetBlockId(vector<size_t>& v) const
{
	v.clear();
	size_t nc = (m_format & matrixFormatRowMajor) ? m_numRows : m_numCols;
	if (m_compPos) for (size_t j=0; j<nc; ++j) v.push_back(m_compPos[j]);
}

template<class ElemType>
void BaseMatrixStorage<ElemType>::TransposeFrom(const BaseMatrixStorage<ElemType>& bms, bool hdr)
{
	if (bms.IsDenseFormat())
	{
		Init(bms.m_format);
		Create(bms.m_numCols, bms.m_numRows); if (IsEmpty()) return;

		bool rmf = IsRowMajor();
		size_t nc = (m_format & matrixFormatRowMajor) ? bms.m_numRows : bms.m_numCols;
		size_t nr = (m_format & matrixFormatRowMajor) ? bms.m_numCols : bms.m_numRows;

		const ElemType* src = bms.m_pBuffer;
		for (size_t j=0; j<nc; ++j)
		{
			ElemType* po = m_pBuffer + j;
			for (size_t i=0; i<nr; ++i) { *po = *src++; po += nc; }
		}
	}
	else if (bms.IsSparseFormat())
	{
		Init(MatrixFormat(bms.m_format ^ matrixFormatRowMajor));
		Create(bms.m_numRows, bms.m_numCols);
		if (!IsEmpty())
		{
			SparseData<ElemType> spd; bms.GetSparseData(spd);
			if (IsRowMajor()) spd.SortByRows(); else spd.SortByCols();
			PutSparseData(spd);
		}
		TransposeHeader();
	}
	else
	{
		Init(bms.m_format);
		Create(bms.m_numCols, bms.m_numRows); if (IsEmpty()) return;

		bool rmf = IsRowMajor();
		size_t nc = (m_format & matrixFormatRowMajor) ? bms.m_numRows : bms.m_numCols;
		size_t nr = (m_format & matrixFormatRowMajor) ? bms.m_numCols : bms.m_numRows;

		Allocate(bms.m_blockCnt*nr);
		for (size_t j=0; j<nc; ++j)
		{
			size_t k = bms.m_compPos[j]; if (k==string::npos) continue;
			const ElemType* p = bms.m_pBuffer + k*nr;
			if (rmf) for (size_t i=0; i<nr; ++i) { if (*p++) PutItem(i,j,p[-1]); }
			else for (size_t i=0; i<nr; ++i) { if (*p++) PutItem(j,i,p[-1]); }
		}
	}
	if (hdr) TransposeHeader();
}

template<class ElemType>
int BaseMatrixStorage<ElemType>::Compare(const BaseMatrixStorage<ElemType>& bms) const
{
	if (bms.m_format!=m_format) return diffFormat;
	if (bms.m_deviceId!=m_deviceId) return diffDevice;
	if (bms.m_numRows!=m_numRows) return diffRows;
	if (bms.m_numCols!=m_numCols) return diffCols;

	size_t n = m_numRows * m_numCols; if (n==0) return 0;
	if (m_format & matrixFormatSparse)
	{
		size_t nc = (m_format & matrixFormatRowMajor) ? m_numRows : m_numCols;
		size_t nr = (m_format & matrixFormatRowMajor) ? m_numCols : m_numRows;

		if (IsSparseFormat())
		{
			int m = (m_compPos ? 1:0) + (bms.m_compPos ? 2:0); if (m==0) return 0;
			if (m==1 || m==2 || memcmp(m_compPos, bms.m_compPos, (nc+1)*sizeof(index_t))) return diffIndex;
			n = m_compPos[nc]; if (n==0) return 0;
			if (memcmp(m_compId, bms.m_compId, n*sizeof(index_t))) return diffIndex;
		}
		else
		{
			if (m_blockCnt!=bms.m_blockCnt) return diffIndex;
			if (m_blockCnt==0) return 0;

			vector<size_t> v1; GetBlockId(v1);
			vector<size_t> v2; bms.GetBlockId(v2);
			const ElemType* src1 = m_pBuffer;
			const ElemType* src2 = bms.m_pBuffer;
			for (size_t j=0; j<v1.size(); ++j)
			{
				int m = (v1[j]==string::npos ? 0:1) + (v2[j]==string::npos ? 0:2);
				if (m==1 || m==2) return diffIndex;
				if (m==0) continue;

				const ElemType* p1 = src1 + v1[j]*nr;
				const ElemType* p2 = src2 + v2[j]*nr;
				if (memcmp(p1, p2, nr*sizeof(ElemType)))
				{
					//for (size_t i=0; i<nr; ++i) cout << " " << float(p1[i]); cout << endl;
					//for (size_t i=0; i<nr; ++i) cout << " " << float(p2[i]); cout << endl;
					return diffData;
				}
			}
			return 0;
		}
	}
	// data
	if (memcmp(m_pBuffer, bms.m_pBuffer, n*sizeof(ElemType))) return diffData;
	return 0;
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//			get / put value
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

template <class ElemType>
ElemType BaseMatrixStorage<ElemType>::GetItem(size_t row, size_t col, size_t offset) const
{
	size_t nr = m_numRows, nc = m_numCols;
	if (m_format & matrixFormatRowMajor) { size_t i = row; row = col; col = i; nr = m_numCols; nc = m_numRows; }

	// dense
	if (IsDenseFormat())
	{
		col += offset/nr; if (col>=nc || row>=nr) return 0;
		return m_pBuffer[col*nr + row];
	}
	// sparse compressed
	col += offset; if (col>=nc || row>=nr) return 0;
	if ((m_format & matrixFormatBlock)==0)
	{
		size_t pos = m_compPos[col];
		for (size_t fin=m_compPos[col+1]; pos<fin; ++pos)
			if (m_compId[pos]==row) return m_pBuffer[pos];
	}
	// sparse block (new)
	else
	{
		size_t i = m_compPos[col];
		return (i==string::npos) ? 0 : m_pBuffer[i*nr + row];
	}
	return 0;
}

template <class ElemType>
void BaseMatrixStorage<ElemType>::PutItem(size_t row, size_t col, ElemType val, size_t offset)
{
	size_t nr = m_numRows, nc = m_numCols;
	if (m_format & matrixFormatRowMajor) { size_t i = row; row = col; col = i; nr = m_numCols; nc = m_numRows; }

	// dense
	if (IsDenseFormat())
	{
		col += offset/nr; if (col>=nc || row>=nr) return;
		m_pBuffer[col*nr + row] = val;
	}
	// sparse compressed
	else if ((m_format & matrixFormatBlock)==0)
	{
		col += offset; if (col>=nc || row>=nr) return;
		size_t pos = m_compPos[col];
		for (size_t fin=m_compPos[col+1]; pos<fin; ++pos)
			if (m_compId[pos]==row) { m_pBuffer[pos] = val; return; }

		size_t n = m_compPos[nc]; if (n==m_buffSize) Allocate(m_buffSize+max(nr,nc));
		for (size_t i=n; i>pos; --i) { m_pBuffer[i] = m_pBuffer[i-1]; m_compId[i] = m_compId[i-1]; }
		for (size_t i=col+1; i<=nc; ++i) ++m_compPos[i];
		m_pBuffer[pos] = val; m_compId[pos] = index_t(row);
	}
	// sparse block (new)
	else
	{
		col += offset; if (col>=nc || row>=nr) return;
		size_t i = m_compPos[col];
		if (i==string::npos)
		{
			if ((m_blockCnt+1)*nr > m_buffSize)
			{
				size_t k = int(0.3*nc + 0.5); if (k==0) ++k;
				Allocate(m_buffSize + k*nr);
			}
			m_compPos[col] = index_t(i = m_blockCnt++);
		}
		m_pBuffer[i*nr + row] = val;
	}
}

template <class ElemType>
ElemType& BaseMatrixStorage<ElemType>::Item(size_t row, size_t col, size_t offset)
{
	size_t nr = m_numRows, nc = m_numCols;
	if (m_format & matrixFormatRowMajor) { size_t i = row; row = col; col = i; nr = m_numCols; nc = m_numRows; }
	col += (IsDenseFormat()) ? offset/nr : offset;
	if (col>=nc || row>=nr) InvalidArgument("Item (%lu,%lu) is out of range [%lu,%lu]", row, col, nr, nc);

	if (IsDenseFormat()) return m_pBuffer[col*nr + row];
	if (IsSparseFormat())
	{
		size_t pos = m_compPos[col];
		for (size_t fin=m_compPos[col+1]; pos<fin; ++pos)
			if (m_compId[pos]==row) return m_pBuffer[pos];

		size_t n = m_compPos[nc]; if (n==m_buffSize) Allocate(m_buffSize+max(nr,nc));
		for (size_t i=n; i>pos; --i) { m_pBuffer[i] = m_pBuffer[i-1]; m_compId[i] = m_compId[i-1]; }
		for (size_t i=col+1; i<=nc; ++i) ++m_compPos[i];
		m_compId[pos] = index_t(row);
		return m_pBuffer[pos];
	}
	size_t i = m_compPos[col];
	if (i==string::npos)
	{
		if ((m_blockCnt+1)*nr > m_buffSize)
		{
			size_t k = int(0.3*nc + 0.5); if (k==0) ++k;
			Allocate(m_buffSize + k*nr);
		}
		m_compPos[col] = index_t(i = m_blockCnt++);
	}
	return m_pBuffer[i*nr + row];
}

template <class ElemType>
void BaseMatrixStorage<ElemType>::GetSparseData(SparseData<ElemType>& spd, size_t offset, size_t len) const
{
	bool rmf = IsRowMajor();
	size_t nc = (rmf) ? m_numRows : m_numCols;
	size_t nr = (rmf) ? m_numCols : m_numRows;
	if (len>0) nc = len;

	spd.Init(m_format);
	if (IsDenseFormat())
	{
		ElemType* p = m_pBuffer + offset;
		size_t n = (nc*nr)/2; spd.reserve(n<16 ? 16:n);
		for (size_t j=0; j<nc; ++j)
			if (rmf) for (size_t i=0; i<nr; ++i) { if (*p++) spd.push_back(ElemItem<ElemType>(j,i,p[-1])); }
			else for (size_t i=0; i<nr; ++i) { if (*p++) spd.push_back(ElemItem<ElemType>(i,j,p[-1])); }
	}
	else if (IsSparseFormat())
	{
		index_t* compPos = m_compPos + offset;
		size_t n = compPos[nc] - compPos[0]; if (n==0) return;
		spd.reserve(n);
		for (size_t j=0; j<nc; ++j)
		{
			size_t ns = compPos[j], ne = compPos[j+1]; if (ns==ne) continue;
			if (rmf) for (size_t k=ns; k<ne; ++k) spd.push_back(ElemItem<ElemType>(j,m_compId[k],m_pBuffer[k]));
			else for (size_t k=ns; k<ne; ++k) spd.push_back(ElemItem<ElemType>(m_compId[k],j,m_pBuffer[k]));
		}
	}
	else
	{
		if (m_blockCnt==0) return;
		spd.reserve(m_blockCnt*nr);
		index_t* blockPos = m_compPos + offset;
		for (size_t j=0; j<nc; ++j)
		{
			size_t k = blockPos[j]; if (k==string::npos) continue;
			ElemType* p = m_pBuffer + k*nr;
			if (rmf) for (size_t i=0; i<nr; ++i) { if (*p++) spd.push_back(ElemItem<ElemType>(j,i,p[-1])); }
			else for (size_t i=0; i<nr; ++i) { if (*p++) spd.push_back(ElemItem<ElemType>(i,j,p[-1])); }
		}
	}
	spd.SetSorted();
}

template <class ElemType>
void BaseMatrixStorage<ElemType>::PutSparseData(const SparseData<ElemType>& spd)
{
	Reset(); if (spd.empty()) return;

	bool rmf = IsRowMajor();
	size_t nc = (rmf) ? m_numRows : m_numCols;
	size_t nr = (rmf) ? m_numCols : m_numRows;
	if (IsDenseFormat())
	{
		if (rmf) for (SparseData<ElemType>::const_iterator i=spd.begin(); i!=spd.end(); ++i)
		{
			if (size_t((*i).row)>m_numRows || size_t((*i).col)>m_numCols)
				RuntimeError("PutSparseData; Item [%d,%d] is out of range (%lu,%lu)", (*i).row, (*i).col, m_numRows, m_numCols);
			m_pBuffer[(*i).row*m_numCols + (*i).col] = (*i).value;
		}
		else for (SparseData<ElemType>::const_iterator i=spd.begin(); i!=spd.end(); ++i)
		{
			if (size_t((*i).row)>m_numRows || size_t((*i).col)>m_numCols)
				RuntimeError("PutSparseData; Item [%d,%d] is out of range (%lu,%lu)", (*i).row, (*i).col, m_numRows, m_numCols);
			m_pBuffer[(*i).col*m_numRows + (*i).row] = (*i).value;
		}
	}
	else if (IsSparseFormat())
	{
		index_t n = 0, k = -1;
		if (spd.size()>m_buffSize) Allocate(spd.size());
		if (rmf) for (SparseData<ElemType>::const_iterator i=spd.begin(); i!=spd.end(); ++i)
		{
			if (size_t((*i).row)>m_numRows || size_t((*i).col)>m_numCols)
				RuntimeError("PutSparseData; Item [%d,%d] is out of range (%lu,%lu)", (*i).row, (*i).col, m_numRows, m_numCols);

			while (k<(*i).row) m_compPos[++k] = n;
			m_pBuffer[n] = (*i).value;
			m_compId[n++] = (*i).col;
		}
		else for (SparseData<ElemType>::const_iterator i=spd.begin(); i!=spd.end(); ++i)
		{
			if (size_t((*i).row)>m_numRows || size_t((*i).col)>m_numCols)
				RuntimeError("PutSparseData; Item [%d,%d] is out of range (%lu,%lu)", (*i).row, (*i).col, m_numRows, m_numCols);

			while (k<(*i).col) m_compPos[++k] = n;
			m_pBuffer[n] = (*i).value;
			m_compId[n++] = (*i).row;
		}
		while (k<nc) m_compPos[++k] = n;
	}
	else
	{
		size_t m = nc/4; m = (m==0) ? nr : m*nr;
		if (rmf) for (SparseData<ElemType>::const_iterator i=spd.begin(); i!=spd.end(); ++i)
		{
			if (size_t((*i).row)>m_numRows || size_t((*i).col)>m_numCols)
				RuntimeError("PutSparseData; Item [%d,%d] is out of range (%lu,%lu)", (*i).row, (*i).col, m_numRows, m_numCols);

			size_t k = m_compPos[(*i).row];
			if (k==string::npos)
			{
				size_t n = m_blockCnt*nr; if (n+nr>m_buffSize) Allocate(n+m);
				m_compPos[(*i).row] = index_t(k = m_blockCnt++);
			}
			m_pBuffer[k*nr + (*i).col] = (*i).value;
		}
		else for (SparseData<ElemType>::const_iterator i=spd.begin(); i!=spd.end(); ++i)
		{
			if (size_t((*i).row)>m_numRows || size_t((*i).col)>m_numCols)
				RuntimeError("PutSparseData; Item [%d,%d] is out of range (%lu,%lu)", (*i).row, (*i).col, m_numRows, m_numCols);

			size_t k = m_compPos[(*i).col];
			if (k==string::npos)
			{
				size_t n = m_blockCnt*nr; if (n+nr>m_buffSize) Allocate(n+m);
				m_compPos[(*i).col] = index_t(k = m_blockCnt++);
			}
			m_pBuffer[k*nr + (*i).row] = (*i).value;
		}
	}
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//			view
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

template<class ElemType>
string BaseMatrixStorage<ElemType>::FormatStr() const
{
	if (IsDenseFormat()) return string("Dense") + (IsRowMajor() ? "Row":"Col");
	return string("Sparse") + (IsBlockFormat() ? "BS":"CS") + (IsRowMajor() ? "R":"C");
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
	if (IsGPU(m_deviceId)) { sprintf_s(sz, sizeof(sz), "  GPU-%d", m_deviceId); s += sz; }
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

		s += string("  buffer   = ") + hex(m_pBuffer) + "\t(" + long(m_buffSize) + ")";
		if (IsDenseFormat()) s += string("\tn=") + long(m_numRows*m_numCols) + "\n";
		else if (IsSparseFormat()) s += string("\tn=") + long(m_compPos ? m_compPos[nc]:0) + "\n";
		else s += string("\tn=") + long(m_blockCnt*nr) + "\n";

		s += string("  compPos  = ") + hex(m_compPos) + "\t(" + long(m_compSize) + ")";
		if (IsBlockFormat()) s += string("\tn=") + long(m_blockCnt) + "\n";
		else s += string("\tn=") + long(m_compPos ? nc+1:0) + "\n";

		s += string("  compId   = ") + hex(m_compId) + "\t(" + long(m_compId ? m_buffSize:0) + ")";
		if (IsSparseFormat()) s += string("\tn=") + long(m_compPos ? m_compPos[nc]:0) + "\n";
		else s += "\n";
	}
	return s;
}

template <class ElemType>
void BaseMatrixStorage<ElemType>::ViewBuffer(ostream& os, const char* fmt, size_t offset, size_t len) const
{
	if (m_numRows*m_numCols == 0) return;
	if (IsDenseFormat()) ViewDense(os,fmt, offset,len);
	else if (IsSparseFormat()) ViewSparse(os,fmt, offset,len);
	else ViewBlock(os,fmt, offset,len);
}

template <class ElemType>
void BaseMatrixStorage<ElemType>::ViewDense(ostream& os, const char* fmt, size_t offset, size_t len) const
{
	size_t nc = (m_format & matrixFormatRowMajor) ? m_numRows : m_numCols;
	size_t nr = (m_format & matrixFormatRowMajor) ? m_numCols : m_numRows;
	offset /= nr; if (len > 0 && (offset + len) < nc) nc = offset + len;

	const ElemType* p = m_pBuffer + offset*nr;
	for (size_t j=offset; j<nc; ++j)
	{
		string s; char sz[32];
		for (size_t i=0; i<nr; ++i)
			{ sprintf_s(sz, sizeof(sz), fmt, float(*p++)); s += sz; }
		os << s << endl;
	}
}

template <class ElemType>
void BaseMatrixStorage<ElemType>::ViewSparse(ostream& os, const char* fmt, size_t offset, size_t len) const
{
	size_t nc = (m_format & matrixFormatRowMajor) ? m_numRows : m_numCols;
	size_t nr = (m_format & matrixFormatRowMajor) ? m_numCols : m_numRows;
	if (len > 0 && (offset + len) < nc) nc = offset + len;

	list<ElemItem<ElemType>> l;
	for (size_t j=offset; j<nc; ++j)
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
void BaseMatrixStorage<ElemType>::ViewBlock(ostream& os, const char* fmt, size_t offset, size_t len) const
{
	size_t nc = (m_format & matrixFormatRowMajor) ? m_numRows : m_numCols;
	size_t nr = (m_format & matrixFormatRowMajor) ? m_numCols : m_numRows;
	if (len > 0 && (offset + len) < nc) nc = offset + len;

	for (size_t j=offset; j<nc; ++j)
	{
		string s; char sz[32];
		if (m_compPos[j]>=0)
		{
			const ElemType* p = m_pBuffer + m_compPos[j]*nr;
			for (size_t i=0; i<nr; ++i) { sprintf_s(sz, sizeof(sz), fmt, float(*p++)); s += sz; }
		}
		else for (size_t i=0; i<nr; ++i) { sprintf_s(sz, sizeof(sz), fmt, 0.0); s += sz; }
		os << s << endl;
	}
}

template <class ElemType>
void BaseMatrixStorage<ElemType>::ViewIds(ostream& os) const
{
	size_t nc = IsRowMajor() ? m_numRows : m_numCols;
	if (IsSparseFormat())
	{
		if (m_compPos)
		{
			size_t k = m_compPos[nc];
			os << "  compPos {"; for (size_t j=0; j<=nc; ++j) os << " " << m_compPos[j]; os << " }" << endl;
			os << "  compId  {"; if (m_compId) for (size_t j=0; j<k; ++j) os << " " << m_compId[j]; os << " }" << endl;
		}
		else os << "  compPos {}" << endl
				<< "  compId {}" << endl;
	}
	else if (IsBlockFormat())
	{
		if (m_compPos)
		{
			os << "  compPos {";
			for (size_t j=0; j<nc; ++j)
				if (m_compPos[j]==string::npos) os << " -";
				else os << " " << m_compPos[j];
			os << " }  " << m_blockCnt << endl;
		}
		else os << "  compPos {}  " << m_blockCnt << endl;
	}
	else os << "  compPos {}" << endl
			<< "  compId {}" << endl;
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
	//void Init() { m_sob->SetZeros(); }
	void Reset() { m_sob->Reset(); ResizeBack(); }

	void Assign(size_t rows, size_t cols, ElemType* p, int flags=matrixFlagNone);
	void Assign(const BaseMatrix<ElemType>& mat, bool shallow = false);
	void Resize(size_t rows, size_t cols);
	void ResizeBack() { m_numRows = m_sob->GetNumRows(); m_numCols = m_sob->GetNumCols(); m_sliceOffset = 0; }
	void Reshape(size_t rows, size_t cols);

	string FormatStr() const { return m_sob->FormatStr(); }
	MatrixFormat GetFormat() const { return m_sob->GetFormat(); }
	device_t GetDeviceId() const { return m_sob->GetDeviceId(); }

	size_t GetNumRows() const { return m_numRows; }
	size_t GetNumCols() const { return m_numCols; }
	size_t GetNumStorageRows() const { return m_sob->GetNumRows(); }
	size_t GetNumStorageCols() const { return m_sob->GetNumCols(); }
	size_t GetNumElements() const { return m_numRows * m_numCols; }
	size_t GetDiagSize() const { return m_numRows < m_numCols ? m_numRows : m_numCols; }
	size_t GetItemCount() const { return (IsDenseFormat()) ? m_numRows*m_numCols : m_sob->GetItemCount(); }
	size_t NzCount() const { return GetItemCount(); }

	bool IsDenseFormat() const { return m_sob->IsDenseFormat(); }
	bool IsSparseFormat() const { return m_sob->IsSparseFormat(); }
	bool IsBlockFormat() const { return m_sob->IsBlockFormat(); }
	bool IsColMajor() const { return m_sob->IsColMajor(); }
	bool IsRowMajor() const { return m_sob->IsRowMajor(); }
	bool IsEmpty() const { return m_numRows == 0 || m_numCols == 0; }

	bool HasExternalBuffer() const { return m_sob->HasExternalBuffer(); }
	bool HasOwnBuffer() const { return !m_sob->HasExternalBuffer(); }
	bool IsSlice() const { return (m_numRows!=m_sob->GetNumRows() || m_numCols!=m_sob->GetNumCols() || m_sliceOffset!=0); }

	ElemType* GetBuffer() const { return m_sob->GetBuffer(); }											// dense/sparse
	ElemType* GetData() const { return m_sob->GetBuffer() + m_sliceOffset; }							// dense
	ElemType* GetDataCol(size_t j) const { return m_sob->GetBuffer() + m_sliceOffset + j*m_numRows; }	// dense, col major
	ElemType* GetDataRow(size_t j) const { return m_sob->GetBuffer() + m_sliceOffset + j*m_numCols; }	// dense, row major

	ElemType GetItem(size_t row, size_t col) const { return m_sob->GetItem(row, col, m_sliceOffset); }
	void PutItem(size_t row, size_t col, ElemType val) { m_sob->PutItem(row, col, val, m_sliceOffset); }
	ElemType& Item(size_t row, size_t col) { return m_sob->Item(row,col); }

	void GetSparseData(SparseData<ElemType>& spd) const;
	void PutSparseData(const SparseData<ElemType>& spd);

	BaseMatrix<ElemType>& TransposeTo(BaseMatrix<ElemType>& mat, bool hdr=false) const { mat.m_sob->TransposeFrom(*m_sob.get(),hdr); mat.ResizeBack(); return mat; }

	void CopyToDense(BaseMatrix<ElemType>& mat) const;
	void CopyToSparse(BaseMatrix<ElemType>& mat) const;
	void CopyToBlock(BaseMatrix<ElemType>& mat) const;
	void CopyToFullBlock(BaseMatrix<ElemType>& mat) const;

	size_t CopyToArray(ElemType* p, size_t n) const;		// copy to dense
	ElemType* CopyToArray() const;

	bool IsEqualTo(const BaseMatrix<ElemType>& m, ElemType thresh=1.e-8) const;
	int  Compare(const BaseMatrix<ElemType>& mat) const;

	// sparse format
	size_t GetBlockCount() const { return m_sob->GetBlockCount(); }

	//index_t* GetCompPos() const { return m_sob->GetCompPos(); }
	index_t* GetPrimePos() const { return m_sob->GetCompPos() + m_sliceOffset; }
	index_t* GetCompId() const { return m_sob->GetCompId(); }

	//index_t* GetBlockPos() const { return m_sob->GetBlockPos() + m_sliceOffset; }
	//ElemType* GetNzValues() { return m_sob->GetNzValues(); }

	void SetSlice(size_t offset, size_t len);
	void SetColumnSlice(size_t offset, size_t len);

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
	void ViewBuffer(ostream& os, const char* fmt="%6.2f") const { m_sob->ViewBuffer(os, fmt); }
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
	void VerifyMigratable(const char* func) const
	{
		if (!m_sob.unique())
			LogicError("%s; Cannot migrate the matrix between devices with slice", func);
		if (m_sob->HasExternalBuffer())
			LogicError("%s; Cannot migrate the matrix between devices with external buffer", func);
	}
	// This is needed for Sparse Matrices to ensure they can write to the matrix. Note: writing to slices is not currently supported
	//void VerifyWritable(const char* function) const
	//{
	//	if (m_numRows!=m_sob->GetNumRows() || m_numCols!=m_sob->GetNumCols())
	//		LogicError("%s: Cannot write to the matrix because it is a slice.", function);
	//}
	void VerifySize(size_t rows, size_t cols, const char* proc)
	{
		if (rows!=m_numRows || cols!=m_numCols)
			LogicError("%s; The matrix is [%lu,%lu], but expected [%lu,%lu]", proc, rows, cols, m_numRows, m_numCols);
	}
};

} } }
