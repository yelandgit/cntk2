//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
//#ifdef _WIN32
//#include <crtdefs.h>
//#endif
#include "Math/CPUMatrix.h"
#include "Math/CPUSparseMatrix.h"
#include "gtest/gtest.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

typedef CPUSingleMatrix DenseMatrix;
typedef CPUSingleSparseMatrix SparseMatrix;

class CPUSparseMatrixTests : public ::testing::Test
{
public:
	static void SetUpTestCase() {}
	static void TearDownTestCase() {}
};

static void TestSetValues(MatrixFormat mft)
{
	SparseMatrix m1(mft);
	std::array<float, 12> array = { 0, 2, 0, 4, 0, 6, 0, 8, 0, 10, 0, 12 };
	m1.Assign(3, 4, array.data());
	ASSERT_EQ(m1.GetItemCount(), (mft & matrixFormatBlock ? 12:6));

	cout << "\t\tmatrix " << m1.FormatStr() << endl;

	SparseMatrix m2(m1);
	ASSERT_EQ(m2.GetFormat(), mft);
	ASSERT_TRUE(m2.IsEqualTo(m1));
	ASSERT_EQ(m2(0,0), 0);
	ASSERT_EQ(m2(1,0), 2);
	ASSERT_EQ(m2(2,0), 0);
	ASSERT_EQ(m2(0,1), 4);
	ASSERT_EQ(m2(1,1), 0);
	ASSERT_EQ(m2(2,1), 6);
	ASSERT_EQ(m2(0,2), 0);
	ASSERT_EQ(m2(1,2), 8);
	ASSERT_EQ(m2(2,2), 0);
	ASSERT_EQ(m2(0,3), 10);
	ASSERT_EQ(m2(1,3), 0);
	ASSERT_EQ(m2(2,3), 12);

	m1.Init(MatrixFormat(mft | matrixFormatRowMajor));
	m1.Assign(3, 4, array.data());
	ASSERT_EQ(m1.GetItemCount(), (mft & matrixFormatBlock ? 12:6));
	ASSERT_EQ(m1(0,0), 0);
	ASSERT_EQ(m1(0,1), 4);
	ASSERT_EQ(m1(0,2), 0);
	ASSERT_EQ(m1(0,3), 10);
	ASSERT_EQ(m1(1,0), 2);
	ASSERT_EQ(m1(1,1), 0);
	ASSERT_EQ(m1(1,2), 8);
	ASSERT_EQ(m1(1,3), 0);
	ASSERT_EQ(m1(2,0), 0);
	ASSERT_EQ(m1(2,1), 6);
	ASSERT_EQ(m1(2,2), 0);
	ASSERT_EQ(m1(2,3), 12);

	SparseMatrix m3(m1);
	ASSERT_EQ(m3.GetFormat(), mft);
	ASSERT_TRUE(m3.IsEqualTo(m2));
}

TEST_F(CPUSparseMatrixTests, SetValues)
{
	TestSetValues(matrixFormatSparseCSC);
	TestSetValues(matrixFormatSparseBSC);
}

TEST_F(CPUSparseMatrixTests, ColumnSlice)
{
	size_t m = 100;
	size_t n = 50;

	DenseMatrix dm0(m, n);
	SparseMatrix sm0(m, n, matrixFormatSparseCSC);
	sm0.Allocate(m*n);

	RandomSeedFixture rsf;
	dm0.SetUniformRandomValue(-1, 1, rsf.GetSeed());
	foreach_coord (row, col, dm0)
		sm0.PutItem(row, col, dm0(row,col));

	const size_t start = 10;
	const size_t numCols = 20;
	DenseMatrix dm1 = dm0.GetColumnSlice(start, numCols);
	DenseMatrix dm2 = sm0.GetColumnSlice(start, numCols).CopyToDense();
	ASSERT_TRUE(dm1.IsEqualTo(dm2));
}

TEST_F(CPUSparseMatrixTests, MakeFullBlock)
{
	SparseMatrix sm(4,6,0); Random(sm,12);
	DenseMatrix dm(sm.CopyToDense(),true);
	sm.ConvertToFullBlock();
	ASSERT_EQ(sm.GetFormat(), matrixFormatSparseBSC);
	ASSERT_TRUE(sm.IsEqualTo(dm));
}

static void TestDiagonal(MatrixFormat mft)
{
	SparseMatrix m1(mft);

	cout << "\t\tmatrix " << m1.FormatStr() << endl;

	m1.Resize(4, 3);
	m1.PutItem(0, 0, 10);
	m1.PutItem(1, 1, 10);
	m1.PutItem(2, 2, 10);

	SparseMatrix m2(mft);
	m2.Resize(4, 3);
	m2.SetDiagonalValue(10);
	ASSERT_TRUE(m2.IsEqualTo(m1));

	DenseMatrix dm(1,3);
	dm.PutItem(0, 0, 10);
	dm.PutItem(0, 1, 10);
	dm.PutItem(0, 2, 10);

	SparseMatrix m3(mft);
	m3.Resize(4, 3);
	m3.SetDiagonalValue(dm);
	ASSERT_TRUE(m3.IsEqualTo(m1));

	SparseMatrix sm1(4,6,mft); Random(sm1,18);
	DenseMatrix dm1 = sm1.CopyToDense();
	DenseMatrix dm2 = dm1.Diagonal();
	DenseMatrix dm3 = sm1.Diagonal();
	ASSERT_TRUE(dm3.IsEqualTo(dm2));
}

TEST_F(CPUSparseMatrixTests, Diagonal)
{
	TestDiagonal(matrixFormatSparseCSC);
	TestDiagonal(matrixFormatSparseBSC);
}

TEST_F(CPUSparseMatrixTests, MultiplyAndAdd)
{
	const size_t m = 100;
	const size_t n = 50;
	
	DenseMatrix dm0(m, n);
	RandomSeedFixture rsf;
	dm0.SetUniformRandomValue(-1, 1, rsf.GetSeed());

	DenseMatrix dm1(m, n);
	dm1.SetUniformRandomValue(-300, 1, rsf.GetSeed());
	dm1.InplaceTruncateBottom(0);

	SparseMatrix sm1(matrixFormatSparseCSC);
	sm1.SetValue(m, n, dm1.GetData());

	DenseMatrix dm2(m, n);
	dm2.SetUniformRandomValue(-200, 1, rsf.GetSeed());
	dm2.InplaceTruncateBottom(0);

	SparseMatrix sm2(matrixFormatSparseCSC);
	sm2.SetValue(m, n, dm2.GetData());

	// generate SparseBlockCol matrix
	DenseMatrix dmMul(m, m);
	DenseMatrix::MultiplyAndAdd(dm0, false, dm1, true, dmMul);

	SparseMatrix smMul(m, m, 0, matrixFormatSparseBSC);
	SparseMatrix::MultiplyAndAdd(1, dm0, false, sm1, true, smMul);
	foreach_coord(row, col, dmMul)
	{
		ASSERT_TRUE(abs(smMul(row, col) - dmMul(row, col)) < c_epsilonFloatE4);
	}

	DenseMatrix::MultiplyAndAdd(dm0, false, dm2, true, dmMul);
	SparseMatrix::MultiplyAndAdd(1, dm0, false, sm2, true, smMul);
	foreach_coord(row, col, dmMul)
	{
		ASSERT_TRUE(abs(smMul(row, col) - dmMul(row, col)) < c_epsilonFloatE4);
	}
}

TEST_F(CPUSparseMatrixTests, DoGatherColumnsOf)
{
	size_t m = 100;
	size_t n = 50;

	DenseMatrix dm(m, n);
	RandomSeedFixture rsf;
	std::vector<float> idxValue(n);
	for (size_t j=0; j<n; ++j) idxValue[j] = (j % 3) ? float(j) : -1;
	DenseMatrix idx(1, n, idxValue.data());

	SparseMatrix sm1(matrixFormatSparseCSC);
	dm.SetUniformRandomValue(-300, 1, rsf.GetSeed());
	dm.InplaceTruncateBottom(0);
	sm1.SetValue(m, n, dm.GetData());

	SparseMatrix sm2(matrixFormatSparseCSC);
	dm.SetUniformRandomValue(-200, 1, rsf.GetSeed());
	dm.InplaceTruncateBottom(0);
	sm2.SetValue(m, n, dm.GetData());

	DenseMatrix dm1(sm1.CopyToDense(),true);
	DenseMatrix dm2(sm2.CopyToDense(),true);
	dm2.DoGatherColumnsOf(1, dm1, idx, 1);
	sm2.DoGatherColumnsOf(1, sm1, idx, 1);
	ASSERT_TRUE(sm2.IsEqualTo(dm2));
}

///TEST_F(CPUSparseMatrixTests, OneHot)
///{
///	const size_t num_class = 6;
///	DenseMatrix m0(2, 2);
///	m0(0, 0) = 1;
///	m0(0, 1) = 2;
///	m0(1, 0) = 3;
///	m0(1, 1) = 4;
///	
///	vector<size_t> shape(3);
///	shape[0] = num_class; shape[1] = 2; shape[2] = 2;
///
///	SparseMatrix sm(matrixFormatSparseCSC);
///	sm.AssignOneHot(m0, shape, 0);
///	
///	ASSERT_TRUE(sm.NzCount() == 4);
///	ASSERT_TRUE(sm(1, 0) == 1);
///	ASSERT_TRUE(sm(2, 2) == 1);
///	ASSERT_TRUE(sm(3, 1) == 1);
///	ASSERT_TRUE(sm(4, 3) == 1);
///
///	vector<size_t> shape2(3);
///	shape2[0] = 2; shape2[1] = num_class; shape2[2] = 2;
///	SparseMatrix sm2(matrixFormatSparseCSC);
///	sm2.AssignOneHot(m0, shape2, 1);
///
///	ASSERT_TRUE(sm2.NzCount() == 4);
///	ASSERT_TRUE(sm2(2, 0) == 1);
///	ASSERT_TRUE(sm2(4, 1) == 1);
///	ASSERT_TRUE(sm2(7, 0) == 1);
///	ASSERT_TRUE(sm2(9, 1) == 1);
///
///	DenseMatrix dirtyMatrix(2, 2);
///	dirtyMatrix(0, 0) = 1;
///	dirtyMatrix(0, 1) = -1;
///	dirtyMatrix(1, 0) = 7;
///	dirtyMatrix(1, 1) = 4;
///	
///	SparseMatrix sm3(matrixFormatSparseCSC);
///	sm3.AssignOneHot(dirtyMatrix, shape, 0);
///
///	ASSERT_TRUE(sm3.NzCount() == 4);
///	ASSERT_TRUE(sm3(1, 0) == 1);
///	ASSERT_TRUE(sm3(0, 2) == 0);
///	ASSERT_TRUE(sm3(0, 1) == 0);
///	ASSERT_TRUE(sm3(4, 3) == 1);
///}

} } } }
