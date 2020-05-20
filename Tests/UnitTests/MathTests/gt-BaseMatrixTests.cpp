//
// Copyright (c) Microsoft. All rights reserved.
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "Math/BaseMatrix.h"
#include "gtest/gtest.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

class BaseMatrixTest : public ::testing::Test
{
public:
	static void SetUpTestCase() {}
	static void TearDownTestCase() {}
};

static void CheckBaseMatrixStorage(MatrixFormat mft)
{
	BaseMatrixStorage<float> m(mft);
	
	// Create + PutItem
	//m.Create(3,4);
	//cout << endl << m.GetInfo() << endl; m.ViewIds(cout);
	//cout << endl; m.ViewBuffer(cout);

	//m.PutItem(0,1,1); m.PutItem(0,2,2);
	//m.PutItem(1,0,3); m.PutItem(2,0,4);

	//cout << endl << m.GetInfo() << endl; m.ViewIds(cout);
	//cout << endl; m.ViewBuffer(cout);

	// Create from reference
	std::array<float,12> array = { 0, 3, 4, 1, 0, 0, 2, 0, 0, 0, 0, 0 };
	m.Create(3, 4, array.data(), matrixFlagNone);
	cout << endl << m.GetInfo() << endl; m.ViewIds(cout);
	cout << endl; m.ViewBuffer(cout);
}

TEST_F(BaseMatrixTest, BaseMatrixStorage)
{
	//CheckBaseMatrixStorage(matrixFormatDenseCol);
	//CheckBaseMatrixStorage(matrixFormatDenseRow);
	//CheckBaseMatrixStorage(matrixFormatSparseCSC);
	//CheckBaseMatrixStorage(matrixFormatSparseCSR);
	//CheckBaseMatrixStorage(matrixFormatSparseBSC);
	//CheckBaseMatrixStorage(matrixFormatSparseBSR);
}

TEST_F(BaseMatrixTest, ConstructorNoFlags)
{

	BaseMatrix<float> m;
	ASSERT_TRUE(m.IsEmpty());

	m.Resize(2, 3);
	ASSERT_TRUE(!m.IsEmpty());
	ASSERT_EQ(m.GetNumRows(), 2);
	ASSERT_EQ(m.GetNumCols(), 3);
	ASSERT_EQ(m.GetNumElements(), 6);

	m.PutItem(0, 0, 1);
	m.PutItem(1, 2, 2);
	ASSERT_EQ(m.GetItem(0,0), 1);
	ASSERT_EQ(m.GetItem(1,2), 2);

	BaseMatrix<float> m1(m);
	ASSERT_TRUE(m1.IsEqualTo(m));
}

TEST_F(BaseMatrixTest, CharConstructorNoFlags)
{
	BaseMatrix<char> m;
	ASSERT_TRUE(m.IsEmpty());

	m.Resize(2, 3);
	ASSERT_TRUE(!m.IsEmpty());
	ASSERT_EQ(m.GetNumRows(), 2);
	ASSERT_EQ(m.GetNumCols(), 3);
	ASSERT_EQ(m.GetNumElements(), 6);

	m.PutItem(0, 0, 1);
	m.PutItem(1, 2, 2);
	ASSERT_EQ(m.GetItem(0,0), 1);
	ASSERT_EQ(m.GetItem(1,2), 2);

	BaseMatrix<char> m1(m);
	ASSERT_TRUE(m1.IsEqualTo(m));
}

TEST_F(BaseMatrixTest, ShortConstructorNoFlags)
{
	BaseMatrix<short> m;
	ASSERT_TRUE(m.IsEmpty());

	m.Resize(2, 3);
	ASSERT_TRUE(!m.IsEmpty());
	ASSERT_EQ(m.GetNumRows(), 2);
	ASSERT_EQ(m.GetNumCols(), 3);
	ASSERT_EQ(m.GetNumElements(), 6);

	m.PutItem(0, 0, 1);
	m.PutItem(1, 2, 2);
	ASSERT_EQ(m.GetItem(0,0), 1);
	ASSERT_EQ(m.GetItem(1,2), 2);

	BaseMatrix<short> m1(m);
	ASSERT_TRUE(m1.IsEqualTo(m));
}

TEST_F(BaseMatrixTest, CreateFromData)
{
	BaseMatrix<float> m; 
	std::array<float, 6> array = { 1, 2, 3, 4, 5, 6 };

	m.Init(matrixFormatDenseCol);
	m.Assign(2, 3, array.data(), matrixFlagNone);
	ASSERT_EQ(m.GetItem(0,0), 1);
	ASSERT_EQ(m.GetItem(0,1), 3);
	ASSERT_EQ(m.GetItem(0,2), 5);
	ASSERT_EQ(m.GetItem(1,0), 2);
	ASSERT_EQ(m.GetItem(1,1), 4);
	ASSERT_EQ(m.GetItem(1,2), 6);

	m.Init(matrixFormatDenseRow);
	m.Assign(2, 3, array.data(), matrixFlagNone);
	ASSERT_EQ(m.GetItem(0,0), 1);
	ASSERT_EQ(m.GetItem(0,1), 2);
	ASSERT_EQ(m.GetItem(0,2), 3);
	ASSERT_EQ(m.GetItem(1,0), 4);
	ASSERT_EQ(m.GetItem(1,1), 5);
	ASSERT_EQ(m.GetItem(1,2), 6);

	m.Init(matrixFormatSparseCSC);
	m.Assign(2, 3, array.data(), matrixFlagNone);
	ASSERT_EQ(m.GetItem(0,0), 1);
	ASSERT_EQ(m.GetItem(0,1), 3);
	ASSERT_EQ(m.GetItem(0,2), 5);
	ASSERT_EQ(m.GetItem(1,0), 2);
	ASSERT_EQ(m.GetItem(1,1), 4);
	ASSERT_EQ(m.GetItem(1,2), 6);

	m.Init(matrixFormatSparseCSR);
	m.Assign(2, 3, array.data(), matrixFlagNone);
	ASSERT_EQ(m.GetItem(0,0), 1);
	ASSERT_EQ(m.GetItem(0,1), 2);
	ASSERT_EQ(m.GetItem(0,2), 3);
	ASSERT_EQ(m.GetItem(1,0), 4);
	ASSERT_EQ(m.GetItem(1,1), 5);
	ASSERT_EQ(m.GetItem(1,2), 6);

	m.Init(matrixFormatSparseBSC);
	m.Assign(2, 3, array.data(), matrixFlagNone);
	ASSERT_EQ(m.GetItem(0,0), 1);
	ASSERT_EQ(m.GetItem(0,1), 3);
	ASSERT_EQ(m.GetItem(0,2), 5);
	ASSERT_EQ(m.GetItem(1,0), 2);
	ASSERT_EQ(m.GetItem(1,1), 4);
	ASSERT_EQ(m.GetItem(1,2), 6);

	m.Init(matrixFormatSparseBSR);
	m.Assign(2, 3, array.data(), matrixFlagNone);
	ASSERT_EQ(m.GetItem(0,0), 1);
	ASSERT_EQ(m.GetItem(0,1), 2);
	ASSERT_EQ(m.GetItem(0,2), 3);
	ASSERT_EQ(m.GetItem(1,0), 4);
	ASSERT_EQ(m.GetItem(1,1), 5);
	ASSERT_EQ(m.GetItem(1,2), 6);
}

TEST_F(BaseMatrixTest, Slice)
{
	BaseMatrix<float> m;
	std::array<float, 12> array = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };

	// Dense, column major
	m.Init(matrixFormatDenseCol);
	m.Assign(3, 4, array.data(), matrixFlagNone);
	ASSERT_EQ(m.GetItem(0,1), 4);
	ASSERT_EQ(m.GetItem(1,0), 2);
	ASSERT_EQ(m.GetItem(1,3), 11);
	ASSERT_EQ(m.GetItem(2,2), 9);

	m.SetSlice(1,2);
	ASSERT_EQ(m.GetItem(0,0), 4);
	ASSERT_EQ(m.GetItem(0,1), 7);
	ASSERT_EQ(m.GetItem(2,0), 6);
	ASSERT_EQ(m.GetItem(2,1), 9);

	m.PutItem(1,1,22);
	ASSERT_EQ(m.GetItem(1,1), 22);
	m.ResizeBack();
	ASSERT_EQ(m.GetItem(1,2), 22);

	// Dense, row major
	m.Init(matrixFormatDenseRow);
	m.Assign(3, 4, array.data(), matrixFlagNone);
	ASSERT_EQ(m.GetItem(1,0), 5);
	ASSERT_EQ(m.GetItem(0,1), 2);
	ASSERT_EQ(m.GetItem(2,2), 11);
	ASSERT_EQ(m.GetItem(1,3), 8);

	m.SetSlice(1,2);
	ASSERT_EQ(m.GetItem(0,0), 5);
	ASSERT_EQ(m.GetItem(0,2), 7);
	ASSERT_EQ(m.GetItem(1,0), 9);
	ASSERT_EQ(m.GetItem(1,2), 11);

	m.PutItem(1,1,22);
	ASSERT_EQ(m.GetItem(1,1), 22);
	m.ResizeBack();
	ASSERT_EQ(m.GetItem(2,1), 22);

	// SparseCSC
	m.Init(matrixFormatSparseCSC);
	m.Assign(3, 4, array.data(), matrixFlagNone);
	ASSERT_EQ(m.GetItem(0,1), 4);
	ASSERT_EQ(m.GetItem(1,0), 2);
	ASSERT_EQ(m.GetItem(1,3), 11);
	ASSERT_EQ(m.GetItem(2,2), 9);

	m.SetSlice(1,2);
	ASSERT_EQ(m.GetItem(0,0), 4);
	ASSERT_EQ(m.GetItem(0,1), 7);
	ASSERT_EQ(m.GetItem(2,0), 6);
	ASSERT_EQ(m.GetItem(2,1), 9);

	m.PutItem(1,1,22);
	ASSERT_EQ(m.GetItem(1,1), 22);
	m.ResizeBack();
	ASSERT_EQ(m.GetItem(1,2), 22);

	// SparseCSR
	m.Init(matrixFormatSparseCSR);
	m.Assign(3, 4, array.data(), matrixFlagNone);
	ASSERT_EQ(m.GetItem(1,0), 5);
	ASSERT_EQ(m.GetItem(0,1), 2);
	ASSERT_EQ(m.GetItem(2,2), 11);
	ASSERT_EQ(m.GetItem(1,3), 8);

	m.SetSlice(1,2);
	ASSERT_EQ(m.GetItem(0,0), 5);
	ASSERT_EQ(m.GetItem(0,2), 7);
	ASSERT_EQ(m.GetItem(1,0), 9);
	ASSERT_EQ(m.GetItem(1,2), 11);

	m.PutItem(1,1,22);
	ASSERT_EQ(m.GetItem(1,1), 22);
	m.ResizeBack();
	ASSERT_EQ(m.GetItem(2,1), 22);

	// SparseBSC
	m.Init(matrixFormatSparseBSC);
	m.Assign(3, 4, array.data(), matrixFlagNone);
	ASSERT_EQ(m.GetItem(0,1), 4);
	ASSERT_EQ(m.GetItem(1,0), 2);
	ASSERT_EQ(m.GetItem(1,3), 11);
	ASSERT_EQ(m.GetItem(2,2), 9);

	m.SetSlice(1,2);
	ASSERT_EQ(m.GetItem(0,0), 4);
	ASSERT_EQ(m.GetItem(0,1), 7);
	ASSERT_EQ(m.GetItem(2,0), 6);
	ASSERT_EQ(m.GetItem(2,1), 9);

	m.PutItem(1,1,22);
	ASSERT_EQ(m.GetItem(1,1), 22);
	m.ResizeBack();
	ASSERT_EQ(m.GetItem(1,2), 22);

	// SparseBSR
	m.Init(matrixFormatSparseBSR);
	m.Assign(3, 4, array.data(), matrixFlagNone);
	ASSERT_EQ(m.GetItem(1,0), 5);
	ASSERT_EQ(m.GetItem(0,1), 2);
	ASSERT_EQ(m.GetItem(2,2), 11);
	ASSERT_EQ(m.GetItem(1,3), 8);

	m.SetSlice(1,2);
	ASSERT_EQ(m.GetItem(0,0), 5);
	ASSERT_EQ(m.GetItem(0,2), 7);
	ASSERT_EQ(m.GetItem(1,0), 9);
	ASSERT_EQ(m.GetItem(1,2), 11);

	m.PutItem(1,1,22);
	ASSERT_EQ(m.GetItem(1,1), 22);
	m.ResizeBack();
	ASSERT_EQ(m.GetItem(2,1), 22);
}

TEST_F(BaseMatrixTest, Assign)
{
	BaseMatrix<float> m1, m2, m3;
	std::array<float, 12> array = { 0, 2, 0, 4, 0, 6, 0, 8, 0, 0, 0, 0 };

	m1.Init(matrixFormatDenseCol);
	m1.Assign(3, 4, array.data(), matrixFlagNone);
	m2.Assign(m1); m3.Assign(m1);
	ASSERT_EQ(m2.Compare(m1), 0);

	m1.Init(matrixFormatSparseCSC);
	m1.Assign(3, 4, array.data(), matrixFlagNone);
	m2.Assign(m1);
	ASSERT_EQ(m2.Compare(m1), 0);
	ASSERT_TRUE(m2.IsEqualTo(m3));

	m1.Init(matrixFormatSparseBSC);
	m1.Assign(3, 4, array.data(), matrixFlagNone);
	m2.Assign(m1);
	ASSERT_EQ(m2.Compare(m1), 0);
	ASSERT_TRUE(m2.IsEqualTo(m3));

	m1.Assign(m3, true);
	ASSERT_TRUE(m1.GetBuffer()==m3.GetBuffer());
}

static void TestCopyTo(MatrixFormat mft)
{
	BaseMatrix<float> m1, m2;
	std::array<float, 12> array = { 0, 2, 0, 4, 0, 6, 0, 8, 0, 0, 0, 0 };

	m1.Init(mft);
	m1.Assign(3, 4, array.data(), matrixFlagNone);
	bool rmf = m1.IsRowMajor();

	cout << "    matrix " << m1.Format() << endl;

	m1.CopyToDense(m2);
	ASSERT_TRUE(m2.GetFormat()== (rmf ? matrixFormatDenseRow : matrixFormatDenseCol));
	ASSERT_TRUE(m2.IsEqualTo(m1));
	m1.SetSlice(1,2); m1.CopyToDense(m2); m2.ResizeBack();
	ASSERT_TRUE(m2.IsEqualTo(m1));

	m1.ResizeBack(); m1.CopyToSparse(m2);
	ASSERT_TRUE(m2.GetFormat()== (rmf ? matrixFormatSparseCSR : matrixFormatSparseCSC));
	ASSERT_TRUE(m2.IsEqualTo(m1));
	m1.SetSlice(1,2); m1.CopyToSparse(m2); m2.ResizeBack();
	ASSERT_TRUE(m2.IsEqualTo(m1));

	m1.ResizeBack(); m1.CopyToBlock(m2);
	ASSERT_TRUE(m2.GetFormat()== (rmf ? matrixFormatSparseBSR : matrixFormatSparseBSC));
	ASSERT_TRUE(m2.IsEqualTo(m1));
	m1.SetSlice(1,2); m1.CopyToSparse(m2); m2.ResizeBack();
	ASSERT_TRUE(m2.IsEqualTo(m1));

	//cout << endl << m1.GetInfo() << endl; m1.ViewData(cout);
	//cout << endl << m2.GetInfo() << endl; m2.ViewIds(cout);
	//cout << endl; m2.ViewData(cout);
}

TEST_F(BaseMatrixTest, CopyToDenseSparseBlock)
{
	TestCopyTo(matrixFormatDenseCol);
	TestCopyTo(matrixFormatDenseRow);
	TestCopyTo(matrixFormatSparseCSC);
	TestCopyTo(matrixFormatSparseCSR);
	TestCopyTo(matrixFormatSparseBSC);
	TestCopyTo(matrixFormatSparseBSR);
}

static void TestMakeFullBlock(MatrixFormat mft)
{
	BaseMatrix<float> m1, m2;
	std::array<float, 12> array = { 0, 2, 0, 4, 0, 6, 0, 8, 0, 0, 0, 0 };

	m1.Init(mft);
	m1.Assign(3, 4, array.data(), matrixFlagNone);
	size_t nref = (m1.IsRowMajor()) ? m1.GetNumRows() : m1.GetNumCols();

	cout << "    matrix " << m1.Format() << endl;

	m1.CopyToFullBlock(m2);
	ASSERT_EQ(m2.GetBlockCount(), nref);
	ASSERT_TRUE(m2.IsEqualTo(m1));

	m1.SetSlice(1,2); m1.CopyToFullBlock(m2);
	ASSERT_EQ(m2.GetBlockCount(), 2);
	ASSERT_TRUE(m2.IsEqualTo(m1));
}

static void TestTransposeTo(MatrixFormat mft)
{
	BaseMatrix<float> m1(mft);
	std::array<float, 12> array = { 0, 2, 0, 4, 5, 6, 0, 8, 9, 0, 0, 0 };
	m1.Assign(3, 4, array.data(), matrixFlagNone);

	cout << "    matrix " << m1.Format() << endl;

	BaseMatrix<float> m2; m1.TransposeTo(m2);
	ASSERT_EQ(m2.GetNumRows(), m1.GetNumCols());
	ASSERT_EQ(m2.GetNumCols(), m1.GetNumRows());
	for (size_t j=0; j<m1.GetNumCols(); ++j)
	for (size_t i=0; i<m1.GetNumRows(); ++i)
		ASSERT_EQ(m1.GetItem(i,j), m2.GetItem(j,i));
}

TEST_F(BaseMatrixTest, TransposeTo)
{
	TestTransposeTo(matrixFormatDenseCol);
	TestTransposeTo(matrixFormatDenseRow);
	TestTransposeTo(matrixFormatSparseCSC);
	TestTransposeTo(matrixFormatSparseCSR);
	TestTransposeTo(matrixFormatSparseBSC);
	TestTransposeTo(matrixFormatSparseBSR);
}

static void TestReshape(MatrixFormat mft)
{
	BaseMatrix<float> m1;
	std::array<float, 12> array = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	m1.Init(mft); m1.Assign(3, 4, array.data(), matrixFlagNone);

	cout << "    matrix " << m1.Format() << endl;

	size_t rows = 4;
	size_t cols = 3;
	BaseMatrix<float> m2;
	m2.Assign(m1); m2.Reshape(rows,cols);
	for (size_t j=0; j<m1.GetNumCols(); ++j)
	for (size_t i=0; i<m1.GetNumRows(); ++i)
		if (mft & matrixFormatRowMajor)
		{
			size_t n = i*m1.GetNumCols() + j;
			ASSERT_EQ(m1.GetItem(i,j), m2.GetItem(n/cols,n%cols));
		}
		else
		{
			size_t n = j*m1.GetNumRows() + i;
			ASSERT_EQ(m1.GetItem(i,j), m2.GetItem(n%rows,n/rows));
		}
}

TEST_F(BaseMatrixTest, Reshape)
{
	TestReshape(matrixFormatDenseCol);
	TestReshape(matrixFormatDenseRow);
	TestReshape(matrixFormatSparseCSC);
	TestReshape(matrixFormatSparseCSR);
	TestReshape(matrixFormatSparseBSC);
	TestReshape(matrixFormatSparseBSR);
}

} } } }