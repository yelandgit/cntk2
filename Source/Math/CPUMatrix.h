//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "BaseMatrix.h"
//#include "../Common/Basics.h"
//#include "../Common/File.h"
//#include "../Math/Matrix.h"
#include "CPURNGHandle.h"
#include "Helpers.h"
//#include <vector>
//#include <stdio.h>
//#include <ctime>
//#include <limits.h>
//#include "QuantizedOperations.h"
//#include "half.hpp"

//#include "GPUMatrix.h"
//#include "CPUSparseMatrix.h"
//#include "GPUSparseMatrix.h"


namespace Microsoft { namespace MSR { namespace CNTK {

template<class> class CPUSparseMatrix;

//double logadd(double x, double y);

// To comply with BLAS libraries matrices are stored in ColMajor. However, by default C/C++/C# use RowMajor
// conversion is need when passing data between CPUMatrix and C++ matrices
template <class ElemType>
class MATH_API CPUMatrix : public BaseMatrix<ElemType>
{
	typedef BaseMatrix<ElemType> Base;

public:
	CPUMatrix(int flags=0) : Base(MatrixFormat(flags)) {}
	CPUMatrix(size_t rows, size_t cols, int flags=0) : Base(MatrixFormat(flags)) { Base::Resize(rows, cols); }
	CPUMatrix(size_t rows, size_t cols, ElemType* p, int flags=matrixFlagNone) { SetValue(rows, cols, p, flags); }
	CPUMatrix(const CPUMatrix<ElemType>& mat, bool shallow) { Assign(mat, shallow); }

	CPUMatrix(const CPUMatrix<ElemType>& mat) { SetValue(mat); }
	CPUMatrix<ElemType>& operator=(const CPUMatrix<ElemType>& mat) { if (&mat!=this) SetValue(mat); return *this; }

	CPUMatrix(CPUMatrix<ElemType>&& mat) { Assign(mat, true); mat.Release(); }
	CPUMatrix<ElemType>& operator=(CPUMatrix<ElemType>&& mat) { if (&mat!=this) { Assign(mat, true); mat.Release(); } return *this; }

public:
	CPUMatrix<ElemType>  GetColumnSlice(size_t start, size_t cols) const;
	CPUMatrix<ElemType>& AssignColumnSlice(const CPUMatrix<ElemType>& fromMatrix, size_t start, size_t cols);
	CPUMatrix<ElemType>& PutColumnSlice(const CPUMatrix<ElemType>& fromMatrix, size_t start, size_t cols);

//	void CopyColumnsStrided(const CPUMatrix<ElemType>& fromMatrix, size_t numCols, size_t srcNumColsStride, size_t destNumColsStride);
//
//	CPUMatrix<ElemType> Diagonal() const;
//	CPUSparseMatrix<ElemType> CopyToSparse() const { CPUSparseMatrix<ElemType> sm; Base::CopyToSparse(sm); return sm; }
//	CPUSparseMatrix<ElemType> CopyToBlock() const { CPUSparseMatrix<ElemType> sm; Base::CopyToBlock(sm); return sm; }

	ElemType Adagrad(CPUMatrix<ElemType>& gradients, bool needAveMultiplier);
	void FSAdagrad(CPUMatrix<ElemType>& gradients, CPUMatrix<ElemType>& functionValues, ElemType learnRatePerSample, 
										ElemType momentum, ElemType adaWeight, ElemType adaMul, ElemType unitGainFactor);
	void Adam(CPUMatrix<ElemType>& gradients, CPUMatrix<ElemType>& functionValues, ElemType learnRatePerSample,
										ElemType momentum, ElemType adaWeight, ElemType adaMul, ElemType epsilon, ElemType unitGainFactor, bool adamax=false);
	ElemType RmsProp(CPUMatrix<ElemType>& gradients, ElemType RMS_GAMMA, ElemType RMS_WGT_INC, ElemType RMS_WGT_MAX,
										ElemType RMS_WGT_DEC, ElemType RMS_WGT_MIN, bool needAveMultiplier, bool initialized);

	template<typename GradType>
	void AdaDelta(CPUMatrix<GradType>& gradients, CPUMatrix<ElemType>& functionValues, ElemType learningRate, ElemType rho, ElemType epsilon);

	void AdaDeltaFlushTimestamps(size_t cols, ElemType rho, int* timestamps, int currentTimestamp);

	// RequireSize is now the new preferred method of ensuring the correct size inside of the Matrix class. Since Resize will fail if the storage object has
	// multiple views, RequireSize will first check to see if Resize is required. If it is not, then it short-circuits and is a noop. Otherwise, RequireSize
	// will call Resize, which may fail if the matrix has multiple views.
	//void RequireSize(size_t numRows, size_t numCols, bool growOnly = true); // by default we only reallocate if need to grow

	// Resize first checks to ensure that the caller has the authority to call Resize (i.e., it checks to ensure the underlying data is owned by only this matrix), and then
	// actually resizes the underlying matrix, doing any allocation as required.
	//void Resize(size_t numRows, size_t numCols, bool growOnly = true); // by default we only reallocate if need to grow

//	ElemType* CopyToArray() const;									// allocated by the callee but need to be deleted by the caller
//	size_t CopyToArray(ElemType*& copyTo, size_t& currSize) const;	// allocated by the callee but need to be deleted by the caller
//	void CopySection(size_t numRows, size_t numCols, ElemType* dst, size_t colStride) const;

	inline ElemType& operator()(size_t row, size_t col) { return GetData()[ItemPos(row, col)]; }
	inline const ElemType& operator()(size_t row, size_t col) const { return GetData()[ItemPos(row, col)]; }
	inline ElemType GetFirstItem() const { return GetData()[0]; }

	void SetValue(ElemType v);
	void SetValue(const CPUMatrix<ElemType>& mat);
	//void SetValue(const GPUMatrix<ElemType>& mat);
	//void SetValue(const CPUSparseMatrix<ElemType>& mat);
	//void SetValue(const GPUSparseMatrix<ElemType>& mat);
	void SetValue(size_t rows, size_t cols, ElemType* p, int flags = matrixFlagNone);

//	void MaskColumnsValue(const CPUMatrix<char>& columnsMask, ElemType val, size_t numColsPerMaskEntry);
//
//	void SetColumn(const ElemType* colPointer, size_t colInd);
//	void SetColumn(const CPUMatrix<ElemType>& valMat, size_t colInd);
//	void SetColumn(ElemType val, size_t j);

	void SetDiagonalValue(ElemType v);
	void SetDiagonalValue(const CPUMatrix<ElemType>& vector);
	void SetUniformRandomValue(ElemType low, ElemType high, unsigned long seed = USE_TIME_BASED_SEED);
	void SetUniformRandomValue(RNGHandle& rngHandle, ElemType low, ElemType high);
	void SetGaussianRandomValue(RNGHandle& rngHandle, ElemType mean, ElemType stdev);
	void SetGumbelRandomValue(RNGHandle& rngHandle, ElemType loc, ElemType scale);
	void SetGaussianRandomValue(ElemType mean, ElemType sigma, unsigned long seed = USE_TIME_BASED_SEED);
	void SetTruncatedNormalRandomValue(ElemType mean, ElemType sigma, unsigned long seed = USE_TIME_BASED_SEED);
	void SetUniformRandomMask(ElemType maskRate, ElemType scaleValue, RNGHandle& rngHandle);
	void AddGaussianRandomValue(ElemType mean, ElemType sigma, unsigned long seed = USE_TIME_BASED_SEED);

	CPUMatrix<ElemType> Transpose();
	CPUMatrix<ElemType>& AssignTransposeOf(const CPUMatrix<ElemType>& a);

//	CPUMatrix<ElemType>& DoGatherColumnsOf (ElemType beta, const CPUMatrix<ElemType>& idx, const CPUMatrix<ElemType>& a, ElemType alpha);
//	CPUMatrix<ElemType>& DoScatterColumnsOf(ElemType beta, const CPUMatrix<ElemType>& idx, const CPUMatrix<ElemType>& a, ElemType alpha);

	CPUMatrix<ElemType>& operator+=(ElemType alpha);
	CPUMatrix<ElemType>  operator+(ElemType alpha) const;
	CPUMatrix<ElemType>& AssignSumOf(ElemType alpha, const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& operator+=(const CPUMatrix<ElemType>& a);
	CPUMatrix<ElemType>  operator+(const CPUMatrix<ElemType>& a) const;
	CPUMatrix<ElemType>& AssignSumOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b);

	CPUMatrix<ElemType>& operator-=(ElemType alpha);
	CPUMatrix<ElemType>  operator-(ElemType alpha) const;
	CPUMatrix<ElemType>& AssignDifferenceOf(ElemType alpha, const CPUMatrix<ElemType>& a);
	CPUMatrix<ElemType>& AssignDifferenceOf(const CPUMatrix<ElemType>& a, ElemType alpha);

	CPUMatrix<ElemType>& operator-=(const CPUMatrix<ElemType>& a);
	CPUMatrix<ElemType>  operator-(const CPUMatrix<ElemType>& a) const;
	CPUMatrix<ElemType>& AssignDifferenceOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b);

	CPUMatrix<ElemType>& operator*=(ElemType alpha);
	CPUMatrix<ElemType>  operator*(ElemType alpha) const;
	CPUMatrix<ElemType>& AssignProductOf(ElemType alpha, const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>  operator*(const CPUMatrix<ElemType>& a) const;
	CPUMatrix<ElemType>& AssignProductOf(const CPUMatrix<ElemType>& a, bool transposeA, const CPUMatrix<ElemType>& b, bool transposeB);

	CPUMatrix<ElemType>& operator/=(ElemType alpha);
	CPUMatrix<ElemType>  operator/(ElemType alpha) const;

	CPUMatrix<ElemType>& operator^=(ElemType alpha);		// element-wise power
	CPUMatrix<ElemType>  operator^(ElemType alpha) const;	// element-wise power
	CPUMatrix<ElemType>& AssignElementPowerOf(const CPUMatrix<ElemType>& a, ElemType power);

	CPUMatrix<ElemType>& ElementMultiplyWith(const CPUMatrix<ElemType>& a);
	CPUMatrix<ElemType>& AssignElementProductOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b);
	CPUMatrix<ElemType>& AddElementProductOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b);

	CPUMatrix<ElemType>& AssignElementDivisionOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b);
	CPUMatrix<ElemType>& ElementDivideBy(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& ColumnElementMultiplyWith(const CPUMatrix<ElemType>& a);
	CPUMatrix<ElemType>& RowElementMultiplyWith(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& ColumnElementDivideBy(const CPUMatrix<ElemType>& a);
	CPUMatrix<ElemType>& RowElementDivideBy(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& ElementInverse();
	CPUMatrix<ElemType>& AssignElementInverseOf(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& InplaceSigmoid();
	CPUMatrix<ElemType>& AssignSigmoidOf(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& InplaceLinearRectifierDerivative();
	CPUMatrix<ElemType>& AssignLinearRectifierDerivativeOf(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& InplaceSigmoidDerivative();
	CPUMatrix<ElemType>& AssignSigmoidDerivativeOf(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& InplaceTanh();
	CPUMatrix<ElemType>& AssignTanhOf(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& InplaceAtanh();
	CPUMatrix<ElemType>& AssignAtanhOf(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& InplaceLogSoftmax(bool isColWise);
	CPUMatrix<ElemType>& AssignLogSoftmaxOf(const CPUMatrix<ElemType>& a, bool isColWise);

	CPUMatrix<ElemType>& InplaceHardmax(bool isColWise);
	CPUMatrix<ElemType>& AssignHardmaxOf(const CPUMatrix<ElemType>& a, bool isColWise);

	CPUMatrix<ElemType>& InplaceSqrt();
	CPUMatrix<ElemType>& AssignSqrtOf(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& InplaceExp();
	CPUMatrix<ElemType>& AssignExpOf(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& InplaceLog();
	CPUMatrix<ElemType>& AssignLogOf(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& InplaceLog10();
	CPUMatrix<ElemType>& AssignLog10Of(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& InplaceCosine();
	CPUMatrix<ElemType>& AssignCosineOf(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& InplaceNegativeSine();
	CPUMatrix<ElemType>& AssignNegativeSineOf(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& InplaceTan();
	CPUMatrix<ElemType>& AssignTanOf(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& InplaceAcos();
	CPUMatrix<ElemType>& AssignAcosOf(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& InplaceAsin();
	CPUMatrix<ElemType>& AssignAsinOf(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& InplaceAtan();
	CPUMatrix<ElemType>& AssignAtanOf(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& InplaceCosh();
	CPUMatrix<ElemType>& AssignCoshOf(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& InplaceSinh();
	CPUMatrix<ElemType>& AssignSinhOf(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& InplaceAsinh();
	CPUMatrix<ElemType>& AssignAsinhOf(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& InplaceAbs();
	CPUMatrix<ElemType>& AssignAbsOf(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& InplaceTruncateBottom(ElemType threshold);
	CPUMatrix<ElemType>& AssignTruncateBottomOf(const CPUMatrix<ElemType>& a, ElemType threshold);
	CPUMatrix<ElemType>& InplaceTruncateTop(ElemType threshold);
	CPUMatrix<ElemType>& AssignTruncateTopOf(const CPUMatrix<ElemType>& a, ElemType threshold);
	CPUMatrix<ElemType>& InplaceTruncate(ElemType threshold);
	CPUMatrix<ElemType>& InplaceSoftThreshold(ElemType threshold);

//	CPUMatrix<ElemType>& SetToZeroIfAbsLessThan(ElemType threshold);

	ElemType SumOfAbsElements() const; // sum of all abs(elements)
	ElemType SumOfElements() const;    // sum of all elements
	CPUMatrix<ElemType>& AssignSumOfElements(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& AssignOneHot(const CPUMatrix<ElemType>& a, vector<size_t>& shape, size_t axis);
	CPUMatrix<ElemType>& GatherFromTarget(const CPUMatrix<ElemType>& indices, const CPUMatrix<ElemType>& target, size_t row_elements);
	CPUMatrix<ElemType>& ScatterToIndices(const CPUMatrix<ElemType>& values, const CPUMatrix<ElemType>& indices, size_t row_elements, const CPUMatrix<char>* mask = nullptr);

//	static void VectorSum(const CPUMatrix<ElemType>& a, CPUMatrix<ElemType>& c, bool isColWise);

	void VectorNorm1(CPUMatrix<ElemType>& c, bool isColWise) const;
	CPUMatrix<ElemType>& AssignVectorNorm1of(CPUMatrix<ElemType>& a, bool isColWise);

	void VectorNorm2(CPUMatrix<ElemType>& c, bool isColWise) const;
	CPUMatrix<ElemType>& AssignVectorNorm2of(CPUMatrix<ElemType>& a, bool isColWise);

	void VectorNormInf(CPUMatrix<ElemType>& c, bool isColWise) const;
	CPUMatrix<ElemType>& AssignVectorNormInfOf(CPUMatrix<ElemType>& a, bool isColWise);

//	void AssignNoiseContrastiveEstimation(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, const CPUMatrix<ElemType>& bias,
//											CPUMatrix<ElemType>& tmp, CPUMatrix<ElemType>& c);
//
//	void AssignSoftmaxSum(const CPUMatrix<ElemType>& a, CPUMatrix<ElemType>& softmax);
//
//	void AssignNCEUnnormalizedEval(const CPUMatrix<ElemType>& a,
//									const CPUMatrix<ElemType>& b, const CPUMatrix<ElemType>& bias, CPUMatrix<ElemType>& c);
//
//	CPUMatrix<ElemType>& AssignNCEDerivative(const CPUMatrix<ElemType>& tmp, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, size_t inputIndex, CPUMatrix<ElemType>& c);

	CPUMatrix<ElemType>& AssignInnerProductOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, bool isColWise);
	CPUMatrix<ElemType>& AssignKhatriRaoProductOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b);
	CPUMatrix<ElemType>& AddColumnReshapeProductOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, bool transposeAColumn);

//	CPUMatrix<ElemType>& AddWithScaleOf(ElemType alpha, const CPUMatrix<ElemType>& a);

	ElemType FrobeniusNorm() const;
	CPUMatrix<ElemType>& AssignFrobeniusNormOf(const CPUMatrix<ElemType>& a);

	ElemType MatrixNormInf() const;
	ElemType MatrixNorm1() const;
	ElemType MatrixNorm0() const; // number of non-zero elemets
//	CPUMatrix<ElemType>& AssignSignOf(const CPUMatrix<ElemType>& a);
//	CPUMatrix<ElemType>& AddSignOf(const CPUMatrix<ElemType>& a);

	CPUMatrix<ElemType>& AssignRowSliceValuesOf(const CPUMatrix<ElemType>& a, size_t startIndex, size_t numRows);
	CPUMatrix<ElemType>& AddToRowSliceValuesOf(const CPUMatrix<ElemType>& a, size_t startIndex, size_t numRows);
	CPUMatrix<ElemType>& AddWithRowSliceValuesOf(const CPUMatrix<ElemType>& a, size_t startIndex, size_t numRows);
	// CPUMatrix<ElemType>&  AssignRowStackValuesOf(const std::vector<const CPUMatrix<ElemType>*>& inputMatrices, size_t sliceStartCol, size_t sliceNumCols);

//	CPUMatrix<ElemType>& AssignToRowSliceValuesOf(const CPUMatrix<ElemType>& a, size_t startIndex, size_t numRows);

	CPUMatrix<ElemType>& AssignRepeatOf(const CPUMatrix<ElemType>& a, size_t numRowRepeats, size_t numColRepeats);
	CPUMatrix<ElemType>& AddToRowRepeatValuesOf(const CPUMatrix<ElemType>& a, size_t numRowRepeats);

//	CPUMatrix<ElemType>& AssignPositiveAndShiftedNegSample(const CPUMatrix<ElemType>& a, size_t posNumber, size_t negNumber, size_t shiftNumber);
//	CPUMatrix<ElemType>& AddFoldedPositiveAndShiftedNegSample(const CPUMatrix<ElemType>& a, size_t posNumber, size_t negNumber, size_t shiftNumber);

	void VectorMax(CPUMatrix<ElemType>& maxIndexes, CPUMatrix<ElemType>& maxValues, bool isColWise, int topK = 1) const;
	void VectorMin(CPUMatrix<ElemType>& minIndexes, CPUMatrix<ElemType>& minValues, bool isColWise) const;

//	CPUMatrix<ElemType>& AssignNumOfDiff(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, bool searchInCol = false);

	// sequence training
//	CPUMatrix<ElemType>& DropFrame(const CPUMatrix<ElemType>& label, const CPUMatrix<ElemType>& gamma, const ElemType& threshhold);
//	CPUMatrix<ElemType>& AssignSequenceError(ElemType hsmoothingWeight, const CPUMatrix<ElemType>& label, const CPUMatrix<ElemType>& dnnoutput, const CPUMatrix<ElemType>& gamma, ElemType alpha);
//	CPUMatrix<ElemType>& AssignCTCScore(const CPUMatrix<ElemType>& prob, CPUMatrix<ElemType>& alpha, CPUMatrix<ElemType>& beta, const CPUMatrix<ElemType>& phoneSeq, const CPUMatrix<ElemType>& phoneBoundary, CPUMatrix<ElemType>& totalScore, const vector<size_t>& uttMap, const vector<size_t> & uttBeginFrame, const vector<size_t> & uttFrameNum, const vector<size_t> & uttPhoneNum, size_t samplesInRecurrentStep, size_t maxFrameNum, size_t blankTokenId, const int delayConstraint, bool isColWise);

//	void Print(const char* matrixName, ptrdiff_t rowStart, ptrdiff_t rowEnd, ptrdiff_t colStart, ptrdiff_t colEnd) const;
//	void Print(const char* matrixName = nullptr) const; // print whole matrix. can be expensive
//
//	void ReadFromFile(FILE* f, const char* matrixName); // matrixName is used to verify that correct matrix is read.
//	void WriteToFile(FILE* f, const char* matrixName);  // matrixName is used to verify that correct matrix is read.
//
//	CPUMatrix<ElemType>& AssignPackedConvolutionInput(const CPUMatrix<ElemType>& inputSubBatch,
//														const size_t inputWidth, size_t inputHeight, size_t inputChannels,
//														const size_t outputWidth, size_t outputHeight, size_t outputChannels,
//														const size_t kernelWidth, size_t kernelHeight, size_t horizontalSubsample, size_t verticalSubsample,
//														bool zeroPadding = false);
//	CPUMatrix<ElemType>& UnpackConvolutionInput(CPUMatrix<ElemType>& inputSubBatch,
//												const size_t inputWidth, size_t inputHeight, size_t inputChannels,
//												const size_t outputWidth, size_t outputHeight, size_t outputChannels,
//												const size_t kernelWidth, size_t kernelHeight, size_t horizontalSubsample, size_t verticalSubsample,
//												bool zeroPadding = false) const;
//	CPUMatrix<ElemType>& AssignMaxPoolingResult(const CPUMatrix<ElemType>& inputBatch, size_t channels,
//												const size_t inputWidth, size_t inputHeight, size_t inputSizePerSample,
//												const size_t outputWidth, size_t outputHeight, size_t outputSizePerSample,
//												const size_t windowWidth, size_t windowHeight, size_t horizontalSubsample, size_t verticalSubsample);
//	CPUMatrix<ElemType>& AddMaxPoolingGradient(const CPUMatrix<ElemType>& outputGradientBatch, const CPUMatrix<ElemType>& inputBatch, const CPUMatrix<ElemType>& outputBatch,
//												const size_t channels,
//												const size_t inputWidth, size_t inputHeight, size_t inputSizePerSample,
//												const size_t outputWidth, size_t outputHeight, size_t outputSizePerSample,
//												const size_t windowWidth, size_t windowHeight, size_t horizontalSubsample, size_t verticalSubsample);
//	CPUMatrix<ElemType>& AssignAveragePoolingResult(const CPUMatrix<ElemType>& inputBatch, size_t channels,
//													const size_t inputWidth, size_t inputHeight, size_t inputSizePerSample,
//													const size_t outputWidth, size_t outputHeight, size_t outputSizePerSample,
//													const size_t windowWidth, size_t windowHeight, size_t horizontalSubsample, size_t verticalSubsample);
//	CPUMatrix<ElemType>& AddAveragePoolingGradient(const CPUMatrix<ElemType>& outputGradientBatch,
//													const size_t channels,
//													const size_t inputWidth, size_t inputHeight, size_t inputSizePerSample,
//													const size_t outputWidth, size_t outputHeight, size_t outputSizePerSample,
//													const size_t windowWidth, size_t windowHeight, size_t horizontalSubsample, size_t verticalSubsample);
//
//	void ConvolutionForward(const CPUMatrix<ElemType>& kernel, const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIwht,
//							const CPUMatrix<int>& mpRowRun, const CPUMatrix<int>& runs, CPUMatrix<ElemType>& output) const;
//	void ConvolutionBackwardData(const CPUMatrix<ElemType>& kernel, const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIwht,
//									const CPUMatrix<int>& mpRowRun, const CPUMatrix<int>& runs, CPUMatrix<ElemType>& grad) const;
//	void ConvolutionBackwardKernel(const CPUMatrix<ElemType>& in, const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIwht,
//									const CPUMatrix<int>& mpRowRun, const CPUMatrix<int>& runs, CPUMatrix<ElemType>& kernelGrad) const;
//
//	void UnrollConvolutionInput(size_t unrollCols, size_t mapOutSize, const CPUMatrix<int>& mpRowCol,
//								const CPUMatrix<int>& mpRowRun, const CPUMatrix<int>& runs, CPUMatrix<ElemType>& output) const;
//	void UnrollConvolutionOutput(size_t unrollCols, size_t mapInCount, size_t mapOutCount, const CPUMatrix<int>& mpRowCol,
//									const CPUMatrix<int>& mpRowRun, const CPUMatrix<int>& runs, CPUMatrix<ElemType>& output) const;
//	void UnrollConvolutionInputForKernelBackprop(size_t mapOutSize, const CPUMatrix<int>& mpRowCol,
//													const CPUMatrix<int>& mpRowRun, const CPUMatrix<int>& runs, CPUMatrix<ElemType>& output) const;
//
//	void MaxPoolingForward(const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIndices, const CPUMatrix<int>& indices, CPUMatrix<ElemType>& output) const;
//	void MaxPoolingBackward(const CPUMatrix<ElemType>& out, const CPUMatrix<ElemType>& in,
//							const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIndices, const CPUMatrix<int>& indices,
//							CPUMatrix<ElemType>& grad, bool accumulateGradient) const;
//
//	void MaxROIPoolingForward(size_t numRois, size_t numImg, size_t channels, size_t width, size_t height,
//								const size_t pooledWidth, size_t pooledHeight, const CPUMatrix<ElemType>& roiData, CPUMatrix<ElemType>& output, CPUMatrix<ElemType>& argmax, double spatialScale) const;
//
//	void MaxROIPoolingBackward(size_t numRois, size_t numImg, size_t channels, size_t width, size_t height,
//								const size_t pooledWidth, size_t pooledHeight, const CPUMatrix<ElemType>& roiData, CPUMatrix<ElemType>& grad, CPUMatrix<ElemType>& argmax, double spatialScale) const;
//
//	void MaxUnpooling(const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIndices, const CPUMatrix<int>& indices, const CPUMatrix<ElemType>& poolInput, CPUMatrix<ElemType>& input) const;
//
//	void AveragePoolingForward(const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIndices, const CPUMatrix<int>& indices, CPUMatrix<ElemType>& output, bool poolIncludePad) const;
//	void AveragePoolingBackward(const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIndices, const CPUMatrix<int>& indices,
//								CPUMatrix<ElemType>& grad, bool poolIncludePad, bool accumulateGradient) const;
//
//	template<class StatType>
//	void BatchNormalizationForward(const CPUMatrix<StatType>& scale, const CPUMatrix<StatType>& bias, bool inferenceOnly, double expAvgFactor, double blendFactor, CPUMatrix<StatType>& runMean, CPUMatrix<StatType>& runVariance,
//									CPUMatrix<ElemType>& out, double epsilon, CPUMatrix<StatType>& saveMean, CPUMatrix<StatType>& saveInvStdDev) const;
//
//	template<class StatType>
//	void BatchNormalizationBackward(const CPUMatrix<ElemType>& in, CPUMatrix<ElemType>& grad, const CPUMatrix<StatType>& scale, double blendFactor, const CPUMatrix<StatType>& saveMean, const CPUMatrix<StatType>& saveInvStdDev,
//									CPUMatrix<StatType>& scaleGrad, CPUMatrix<StatType>& biasGrad) const;

public:
	// This functions do not depend on <ElemType>, i.e. you can call them on any <ElemType>
//	static int SetNumThreads(int numThreads);
//	static int GetMaxNumThreads();

	enum OptimizationFlag
	{
		OPT_EVAL_WITH_MKL = 1, // using Intel MKL functions for evaluation performance
	};
	//static void SetOptimizationFlags(int flags);
	//static int  GetOptimizationFlags();

	static void SetCompatibleMode();

	// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
	//		static BLAS functions
	// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

public:
	static void Multiply(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c);
	static void Multiply(const CPUMatrix<ElemType>& a, bool transposeA, const CPUMatrix<ElemType>& b, bool transposeB, CPUMatrix<ElemType>& c);
	static void MultiplyAndAdd(const CPUMatrix<ElemType>& a, bool transposeA, const CPUMatrix<ElemType>& b, bool transposeB, CPUMatrix<ElemType>& c);
	static void MultiplyAndWeightedAdd(ElemType alpha, const CPUMatrix<ElemType>& a, bool transposeA, const CPUMatrix<ElemType>& b, bool transposeB, ElemType beta, CPUMatrix<ElemType>& c);
	static void Multiply1x1AndWeightedAdd(ElemType alpha, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, ElemType beta, CPUMatrix<ElemType>& c);
	//static void MultiplyAndWeightedAdd(ElemType alpha, const CPUMatrix<ElemType>& a, bool transposeA, const CPUMatrix<ElemType>& b, bool transposeB, ElemType beta, CPUMatrix<ElemType>& c, shared_ptr<QuantizedMultiplier<ElemType>> pQuantizedMultiplier=nullptr);
	static void ColumnwiseScaleAndWeightedAdd(ElemType alpha, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& v, ElemType beta, CPUMatrix<ElemType>& c);

	static void Scale(ElemType alpha, CPUMatrix<ElemType>& a);
	//static void Scale(CPUMatrix<ElemType> alpha, CPUMatrix<ElemType>& a);	// alpha must be 1x1
	static void Scale(ElemType alpha, const CPUMatrix<ElemType>& a, CPUMatrix<ElemType>& c);

	static void ScaleAndAdd(ElemType alpha, const CPUMatrix<ElemType>& a, CPUMatrix<ElemType>& c);
	static void AddScaledDifference(ElemType alpha, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c);
	static void AssignScaledDifference(ElemType alpha, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c);
	//static void AddScaledDifference(const CPUMatrix<ElemType>& alpha, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c);    // alpha must be 1x1
	//static void AssignScaledDifference(const CPUMatrix<ElemType>& alpha, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c); // alpha must be 1x1

	static void InnerProduct(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c, bool isColWise);
	static ElemType InnerProductOfMatrices(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b);
	static void ElementWisePower(ElemType alpha, const CPUMatrix<ElemType>& a, CPUMatrix<ElemType>& c);
	static void BatchMatMul(ElemType beta, const CPUMatrix<ElemType>& a, bool transposeA, int m, const CPUMatrix<ElemType>& b, bool transposeB, int n, CPUMatrix<ElemType>& c, bool isColWise);

//	static void AddElementToElement(ElemType beta, const CPUMatrix<ElemType>& a, size_t ai, size_t aj, CPUMatrix<ElemType>& c, size_t ci, size_t cj);
//	static void MinusOneAt(CPUMatrix<ElemType>& c, size_t position);

//	static bool AreEqual(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, ElemType threshold = 1e-8);
//
//	static void SVD(const CPUMatrix<ElemType>& A, CPUMatrix<ElemType>& SIGMA, CPUMatrix<ElemType>& U, CPUMatrix<ElemType>& VT, CPUMatrix<ElemType>& W);
//	static void TensorShuffleScaleAndAdd(ElemType keepWeight, const CPUMatrix<ElemType>& a, size_t D, size_t S, size_t M, size_t K, size_t T, ElemType scaleFactor, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c);


//	void TensorOp(ElemType beta, const CPUMatrix<ElemType>& a, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
//					const std::array<size_t, 2>& offsets,
//					const SmallVector<size_t>& regularOpDims, const std::array<SmallVector<ptrdiff_t>, 2>& regularStrides,
//					const SmallVector<size_t>& reducingOpDims, const std::array<SmallVector<ptrdiff_t>, 2>& reducingStrides);
//	void TensorOp(ElemType beta, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
//					const std::array<size_t, 3>& offsets,
//					const SmallVector<size_t>& regularOpDims, const std::array<SmallVector<ptrdiff_t>, 3>& regularStrides,
//					const SmallVector<size_t>& reducingOpDims, const std::array<SmallVector<ptrdiff_t>, 3>& reducingStrides);
//	void TensorOp(ElemType beta, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, const CPUMatrix<ElemType>& c, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
//					const std::array<size_t, 4>& offsets,
//					const SmallVector<size_t>& regularOpDims, const std::array<SmallVector<ptrdiff_t>, 4>& regularStrides,
//					const SmallVector<size_t>& reducingOpDims, const std::array<SmallVector<ptrdiff_t>, 4>& reducingStrides);
//
//	int Argmin() const;
//	int Argmax() const;
//	int ArgOp(ElementWiseOperator reductionOp) const;
//
//	void TensorArgOp(const CPUMatrix<ElemType>& a, ElementWiseOperator reductionOp,
//						const std::array<size_t, 2>& offsets,
//						const SmallVector<size_t>& regularOpDims, const std::array<SmallVector<ptrdiff_t>, 2>& regularStrides,
//						const SmallVector<size_t>& reducingOpDims, const std::array<SmallVector<ptrdiff_t>, 2>& reducingStrides);

	static CPUMatrix<ElemType> Ones(size_t rows, size_t cols);
	static CPUMatrix<ElemType> Zeros(size_t rows, size_t cols);
	static CPUMatrix<ElemType> Eye(size_t rows);
	static CPUMatrix<ElemType> RandomUniform(size_t rows, size_t cols, ElemType low, ElemType high, unsigned long seed = USE_TIME_BASED_SEED);
	static CPUMatrix<ElemType> RandomGaussian(size_t rows, size_t cols, ElemType mean, ElemType sigma, unsigned long seed = USE_TIME_BASED_SEED);

	// return true if v is an element in matrix c
//	static bool HasElement(const CPUMatrix<ElemType>& a, ElemType v = 0.0);
//
//public:
//	CPUMatrix<ElemType>& AssignElementProductOfWithShiftNeg(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, size_t shift, size_t negnumber);
//	static void InnerProductWithShiftNeg(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c, bool isColWise, size_t shift, size_t negnumber);
//	// extract out a row from a, assign it to [this].
//	CPUMatrix<ElemType>& GetARowByIndex(const CPUMatrix<ElemType>& a, size_t index);
//	static void ConductRowElementMultiplyWithShift(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c, size_t shift, bool bFirstmatrixfixed);
//	CPUMatrix<ElemType>& AssignElementProductOfWithShift(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, size_t shift);
//
//public:
//	friend File& operator >> (File& fis, CPUMatrix<ElemType>& us)
//	{
//		fis.GetMarker("BMAT");
//		size_t elsize; fis >> elsize;
//		if (sizeof(ElemType) != elsize)
//			RuntimeError("Template argument size doesn't match those in file");
//
//		int format;
//		std::string matrixName;
//		size_t numRows, numCols;
//		fis >> matrixName >> format >> numRows >> numCols;
//		ElemType* p = new ElemType[numRows * numCols];
//		for (size_t i=0; i<numRows*numCols; ++i) fis >> p[i];
//		us.SetValue(numRows, numCols, p, matrixFlagNone);
//		delete[] p;
//		fis.GetMarker("EMAT");
//
//		return fis;
//	}
//	friend File& operator << (File& fos, const CPUMatrix<ElemType>& us)
//	{
//		fos.PutMarkerBegin("BMAT");
//		fos << sizeof(ElemType) << L"noname" << int(us.GetFormat())
//				<< us.m_numRows << us.m_numCols << eoln;
//		//for (size_t i = 0; i < us.GetNumElements(); ++i) fos << us.Data()[i];
//		ElemType* p = us.Data();
//		for (size_t j=0; j<us.GetNumRows(); ++j)
//		{
//			for (size_t j=0; j<us.GetNumCols(); ++j) fos << *p++;
//			fos << eoln;
//		}
//		fos.PutMarkerEnd("EMAT");
//		return fos;
//	}
//
//public:
//	ElemType LogSumOfElements() const;
//
//	std::string str() const;
//	void view(std::ostream& os) const;
//
//public:
//	// for RCRF
//	static void RCRFBackwardCompute(const CPUMatrix<ElemType>& alpha, CPUMatrix<ElemType>& beta,
//									const CPUMatrix<ElemType>& lbls,
//									const CPUMatrix<ElemType>& pair_scores);
//	static void _rcrfBackwardCompute(size_t t, size_t k, const CPUMatrix<ElemType>& alpha,
//										CPUMatrix<ElemType>& beta,
//										const CPUMatrix<ElemType>& pair_scores);
//
//	static void RCRFTransGrdCompute(const CPUMatrix<ElemType>& lbls,
//									const CPUMatrix<ElemType>& alpha,
//									const CPUMatrix<ElemType>& beta,
//									const CPUMatrix<ElemType>& pair_scores,
//									CPUMatrix<ElemType>& grd);
//
//	static void _rcrfTransGrdCompute(size_t i,
//										const CPUMatrix<ElemType>& lbls,
//										const CPUMatrix<ElemType>& alpha,
//										const CPUMatrix<ElemType>& beta,
//										const CPUMatrix<ElemType>& pair_scores,
//										CPUMatrix<ElemType>& grd,
//										const size_t tPos // position
//										);

protected:
	inline size_t ItemPos(size_t row, size_t col) const { return IsRowMajor() ? row*m_numCols + col : col*m_numRows + row; }
	inline size_t ColumnPos(size_t col) const { return col*m_numRows; }
	inline size_t RowPos(size_t row) const { return row*m_numCols; }

private:
	void ScatterValues(ElemType* indices, ElemType* value, ElemType* data, ElemType alpha, size_t num_indices, size_t rows, size_t cols, size_t indices_step = 1);
	void ScatterValues(ElemType* indices, ElemType* value, ElemType* data, ElemType alpha, size_t num_indices, size_t rows, size_t cols, char* mask, size_t numElemsPerMaskEntry, size_t indices_step = 1);

//private:
//	static int		m_optimizationFlags;
};

typedef CPUMatrix<half> CPUHalfMatrix;
typedef CPUMatrix<float> CPUSingleMatrix;
typedef CPUMatrix<double> CPUDoubleMatrix;

//template<typename ElemType>
//void CPUMatrixTensorOpImpl(ElemType beta, const CPUMatrix<ElemType>& a, CPUMatrix<ElemType>& o, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
//	const array<size_t, 2>& offsets,
//	const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
//	const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides);
//
//template<typename ElemType>
//void CPUMatrixTensorOpImpl(ElemType beta, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& o, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
//	const array<size_t, 3>& offsets,
//	const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 3>& regularStrides,
//	const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 3>& reducingStrides);
//
//template<typename ElemType>
//void CPUMatrixTensorOpImpl(ElemType beta, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, const CPUMatrix<ElemType>& c, CPUMatrix<ElemType>& o, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
//	const array<size_t, 4>& offsets,
//	const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 4>& regularStrides,
//	const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 4>& reducingStrides);
//
//template<typename ElemType>
//void CPUMatrixTensorArgOpImpl(const CPUMatrix<ElemType>& a, CPUMatrix<ElemType>& o, ElementWiseOperator reductionOp,
//	const array<size_t, 2>& offsets,
//	const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
//	const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides);

} } }
