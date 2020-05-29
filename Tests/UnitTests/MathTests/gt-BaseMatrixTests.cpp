//
// Copyright (c) Microsoft. All rights reserved.
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "Math/BaseMatrix.h"
#include "gtest/gtest.h"
#include <algorithm>

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

class BaseMatrixTests : public ::testing::Test
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

//TEST_F(BaseMatrixTests, BaseMatrixStorage)
//{
//	CheckBaseMatrixStorage(matrixFormatDenseCol);
//	CheckBaseMatrixStorage(matrixFormatDenseRow);
//	CheckBaseMatrixStorage(matrixFormatSparseCSC);
//	CheckBaseMatrixStorage(matrixFormatSparseCSR);
//	CheckBaseMatrixStorage(matrixFormatSparseBSC);
//	CheckBaseMatrixStorage(matrixFormatSparseBSR);
//}

TEST_F(BaseMatrixTests, ConstructorNoFlags)
{

	BaseMatrix<float> m;
	ASSERT_TRUE(m.IsEmpty());

	m.Resize(2, 3);
	ASSERT_TRUE(!m.IsEmpty());
	ASSERT_EQ(m.GetNumRows(), 2);
	ASSERT_EQ(m.GetNumCols(), 3);
	ASSERT_EQ(m.GetItemCount(), 6);

	m.PutItem(0, 0, 1);
	m.PutItem(1, 2, 2);
	ASSERT_EQ(m.GetItem(0,0), 1);
	ASSERT_EQ(m.GetItem(1,2), 2);

	BaseMatrix<float> m1(m);
	ASSERT_TRUE(m1.IsEqualTo(m));
}

TEST_F(BaseMatrixTests, CharConstructorNoFlags)
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

TEST_F(BaseMatrixTests, ShortConstructorNoFlags)
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

static void CheckAssign(MatrixFormat mft)
{
	BaseMatrix<float> m; 
	std::array<float,8> array = { 1, 2, 3, 4, 5, 6, 7, 8 };

	m.Init(mft);
	m.Assign(4, 2, array.data(), matrixFlagNone);
	cout << endl; m.ViewData(cout);

	m.Init(mft);
	m.Assign(4, 2, array.data(), matrixFlagRowMajor);
	cout << endl; m.ViewData(cout);

	m.Init(MatrixFormat(mft | matrixFormatRowMajor));
	m.Assign(4, 2, array.data(), matrixFlagNone);
	cout << endl; m.ViewData(cout);

	m.Init(MatrixFormat(mft | matrixFormatRowMajor));
	m.Assign(4, 2, array.data(), matrixFlagRowMajor);
	cout << endl; m.ViewData(cout);
}

static void TestCreateFromData(MatrixFormat mft)
{
	if ((mft & matrixFormatSparse)==0) cout << "\t\tmatrix DenseCol/Row" << endl;
	else if (mft & matrixFormatBlock) cout << "\t\tmatrix SparseBSC/BSR" << endl;
	else cout << "\t\tmatrix SparseCSC/CSR" << endl;

	BaseMatrix<float> m;
	std::array<float, 6> array = { 1, 2, 3, 4, 5, 6 };

	m.Init(mft);
	m.Assign(3, 2, array.data(), matrixFlagNone);
	ASSERT_EQ(m.GetItem(0,0), 1);
	ASSERT_EQ(m.GetItem(1,0), 2);
	ASSERT_EQ(m.GetItem(2,0), 3);
	ASSERT_EQ(m.GetItem(0,1), 4);
	ASSERT_EQ(m.GetItem(1,1), 5);
	ASSERT_EQ(m.GetItem(2,1), 6);

	m.Init(mft);
	m.Assign(3, 2, array.data(), matrixFlagRowMajor);
	ASSERT_EQ(m.GetItem(0,0), 1);
	ASSERT_EQ(m.GetItem(1,0), 3);
	ASSERT_EQ(m.GetItem(2,0), 5);
	ASSERT_EQ(m.GetItem(0,1), 2);
	ASSERT_EQ(m.GetItem(1,1), 4);
	ASSERT_EQ(m.GetItem(2,1), 6);

	m.Init(MatrixFormat(mft | matrixFormatRowMajor));
	m.Assign(3, 2, array.data(), matrixFlagNone);
	ASSERT_EQ(m.GetItem(0, 0), 1);
	ASSERT_EQ(m.GetItem(0, 1), 4);
	ASSERT_EQ(m.GetItem(1, 0), 2);
	ASSERT_EQ(m.GetItem(1, 1), 5);
	ASSERT_EQ(m.GetItem(2, 0), 3);
	ASSERT_EQ(m.GetItem(2, 1), 6);

	m.Init(MatrixFormat(mft | matrixFormatRowMajor));
	m.Assign(3, 2, array.data(), matrixFlagRowMajor);
	ASSERT_EQ(m.GetItem(0, 0), 1);
	ASSERT_EQ(m.GetItem(0, 1), 2);
	ASSERT_EQ(m.GetItem(1, 0), 3);
	ASSERT_EQ(m.GetItem(1, 1), 4);
	ASSERT_EQ(m.GetItem(2, 0), 5);
	ASSERT_EQ(m.GetItem(2, 1), 6);
}

TEST_F(BaseMatrixTests, CreateFromData)
{
	//CheckAssign(matrixFormatDenseCol);
	//CheckAssign(matrixFormatSparseCSC);
	//CheckAssign(matrixFormatSparseBSC);

	TestCreateFromData(matrixFormatDense);
	TestCreateFromData(matrixFormatSparse);
	TestCreateFromData(matrixFormatSparseBlock);
}

static void TestSetSlice(MatrixFormat mft)
{
	if ((mft & matrixFormatSparse)==0) cout << "\t\tmatrix DenseCol/Row" << endl;
	else if (mft & matrixFormatBlock) cout << "\t\tmatrix SparseBSC/BSR" << endl;
	else cout << "\t\tmatrix SparseCSC/CSR" << endl;

	BaseMatrix<float> m;
	std::array<float, 12> array = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };

	m.Init(mft);
	m.Assign(3, 4, array.data(), matrixFlagNone);
	m.SetSlice(1,2);
	ASSERT_EQ(m.GetItem(0,0), 4);
	ASSERT_EQ(m.GetItem(0,1), 7);
	ASSERT_EQ(m.GetItem(2,0), 6);
	ASSERT_EQ(m.GetItem(2,1), 9);

	m.PutItem(1,1,22);
	ASSERT_EQ(m.GetItem(1,1), 22);
	m.ResizeBack();
	ASSERT_EQ(m.GetItem(1,2), 22);

	m.Init(mft);
	m.Assign(3, 4, array.data(), matrixFlagRowMajor);
	m.SetSlice(1,2);
	ASSERT_EQ(m.GetItem(0,0), 2);
	ASSERT_EQ(m.GetItem(2,0), 10);
	ASSERT_EQ(m.GetItem(0,1), 3);
	ASSERT_EQ(m.GetItem(2,1), 11);

	m.PutItem(1,1,22);
	ASSERT_EQ(m.GetItem(1,1), 22);
	m.ResizeBack();
	ASSERT_EQ(m.GetItem(1,2), 22);

	m.Init(MatrixFormat(mft | matrixFormatRowMajor));
	m.Assign(3, 4, array.data(), matrixFlagNone);
	m.SetSlice(1,2);
	ASSERT_EQ(m.GetItem(0,0), 2);
	ASSERT_EQ(m.GetItem(0,2), 8);
	ASSERT_EQ(m.GetItem(1,0), 3);
	ASSERT_EQ(m.GetItem(1,2), 9);

	m.PutItem(1,1,22);
	ASSERT_EQ(m.GetItem(1,1), 22);
	m.ResizeBack();
	ASSERT_EQ(m.GetItem(2,1), 22);

	m.Init(MatrixFormat(mft | matrixFormatRowMajor));
	m.Assign(3, 4, array.data(), matrixFlagRowMajor);
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

TEST_F(BaseMatrixTests, SetSlice)
{
	TestSetSlice(matrixFormatDense);
	TestSetSlice(matrixFormatSparse);
	TestSetSlice(matrixFormatSparseBlock);
}

TEST_F(BaseMatrixTests, Assign)
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

TEST_F(BaseMatrixTests, CopyToArray)
{
	float data[12];
	BaseMatrix<float> m;
	std::array<float, 12> array = { 0, 2, 0, 4, 0, 6, 0, 8, 0, 10, 0, 12 };

	// dense
	m.Init(matrixFormatDense);
	m.Assign(3, 4, array.data());
	memset(data, 0xff, sizeof(data));
	size_t n = m.CopyToArray(data, sizeof(data)/sizeof(float));
	ASSERT_EQ(n, 12);
	ASSERT_TRUE(memcmp(data, array.data(), n*sizeof(float))==0);

	m.SetSlice(1,2);
	memset(data, 0xff, sizeof(data));
	n = m.CopyToArray(data, sizeof(data)/sizeof(float));
	ASSERT_EQ(n, 6);
	ASSERT_TRUE(memcmp(data, array.data()+3, n*sizeof(float))==0);

	// sparse
	m.Init(matrixFormatSparseCSC);
	m.Assign(3, 4, array.data());
	memset(data, 0xff, sizeof(data));
	n = m.CopyToArray(data, sizeof(data)/sizeof(float));
	ASSERT_EQ(n, 12);
	ASSERT_TRUE(memcmp(data, array.data(), n*sizeof(float))==0);

	m.SetSlice(1,2);
	memset(data, 0xff, sizeof(data));
	n = m.CopyToArray(data, sizeof(data)/sizeof(float));
	ASSERT_EQ(n, 6);
	ASSERT_TRUE(memcmp(data, array.data()+3, n*sizeof(float))==0);

	// block
	m.Init(matrixFormatSparseBSC);
	m.Assign(3, 4, array.data());
	memset(data, 0xff, sizeof(data));
	n = m.CopyToArray(data, sizeof(data)/sizeof(float));
	ASSERT_EQ(n, 12);
	ASSERT_TRUE(memcmp(data, array.data(), n*sizeof(float))==0);

	m.SetSlice(1,2);
	memset(data, 0xff, sizeof(data));
	n = m.CopyToArray(data, sizeof(data)/sizeof(float));
	ASSERT_EQ(n, 6);
	ASSERT_TRUE(memcmp(data, array.data()+3, n*sizeof(float))==0);
}

static void TestCopyTo(MatrixFormat mft)
{
	BaseMatrix<float> m1, m2;
	std::array<float, 12> array = { 0, 2, 0, 4, 0, 6, 0, 8, 0, 0, 0, 0 };

	m1.Init(mft);
	m1.Assign(3, 4, array.data(), matrixFlagNone);
	bool rmf = m1.IsRowMajor();

	cout << "\t\tmatrix " << m1.FormatStr() << endl;

	m1.CopyToDense(m2);
	ASSERT_TRUE(m2.GetFormat()== (rmf ? matrixFormatDenseRow : matrixFormatDenseCol));
	ASSERT_TRUE(m2.IsEqualTo(m1));
	m1.SetSlice(1,2); m1.CopyToDense(m2);
	ASSERT_FALSE(m2.IsSlice());
	ASSERT_TRUE(m2.IsEqualTo(m1));

	m1.ResizeBack(); m1.CopyToSparse(m2);
	ASSERT_TRUE(m2.GetFormat()== (rmf ? matrixFormatSparseCSR : matrixFormatSparseCSC));
	ASSERT_TRUE(m2.IsEqualTo(m1));
	m1.SetSlice(1,2); m1.CopyToSparse(m2);
	ASSERT_FALSE(m2.IsSlice());
	ASSERT_TRUE(m2.IsEqualTo(m1));

	m1.ResizeBack(); m1.CopyToBlock(m2);
	ASSERT_TRUE(m2.GetFormat()== (rmf ? matrixFormatSparseBSR : matrixFormatSparseBSC));
	ASSERT_TRUE(m2.IsEqualTo(m1));
	m1.SetSlice(1,2); m1.CopyToSparse(m2);
	ASSERT_FALSE(m2.IsSlice());
	ASSERT_TRUE(m2.IsEqualTo(m1));
}

TEST_F(BaseMatrixTests, CopyToDenseSparseBlock)
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

	cout << "\t\tmatrix " << m1.FormatStr() << endl;

	m1.CopyToFullBlock(m2);
	ASSERT_EQ(m2.GetBlockCount(), nref);
	ASSERT_TRUE(m2.IsEqualTo(m1));

	m1.SetSlice(1,2); m1.CopyToFullBlock(m2);
	ASSERT_EQ(m2.GetBlockCount(), 2);
	ASSERT_TRUE(m2.IsEqualTo(m1));
}

TEST_F(BaseMatrixTests, MakeFullBlock)
{
	TestMakeFullBlock(matrixFormatDenseCol);
	TestMakeFullBlock(matrixFormatDenseRow);
	TestMakeFullBlock(matrixFormatSparseCSC);
	TestMakeFullBlock(matrixFormatSparseCSR);
	TestMakeFullBlock(matrixFormatSparseBSC);
	TestMakeFullBlock(matrixFormatSparseBSR);
}

static void TestGetPutSparseData(MatrixFormat mft)
{
	std::array<float, 12> array = { 0, 2, 0, 4, 0, 6, 0, 8, 0, 10, 0, 12 };
	BaseMatrix<float> m1(mft);
	m1.Assign(3, 4, array.data());

	cout << "\t\tmatrix " << m1.FormatStr() << endl;

	// whole matrix
	SparseData<float> v; m1.GetSparseData(v);
	if (m1.IsRowMajor())
	{
		ASSERT_EQ(v.size(), 6);
        ASSERT_EQ(v[0], ElemItem<float>(0, 1, 4.f));
        ASSERT_EQ(v[1], ElemItem<float>(0, 3, 10.f));
        ASSERT_EQ(v[2], ElemItem<float>(1, 0, 2.f));
        ASSERT_EQ(v[3], ElemItem<float>(1, 2, 8.f));
        ASSERT_EQ(v[4], ElemItem<float>(2, 1, 6.f));
        ASSERT_EQ(v[5], ElemItem<float>(2, 3, 12.f));
	}
	else
	{
		ASSERT_EQ(v.size(), 6);
        ASSERT_EQ(v[0], ElemItem<float>(1, 0, 2.f));
        ASSERT_EQ(v[1], ElemItem<float>(0, 1, 4.f));
        ASSERT_EQ(v[2], ElemItem<float>(2, 1, 6.f));
        ASSERT_EQ(v[3], ElemItem<float>(1, 2, 8.f));
        ASSERT_EQ(v[4], ElemItem<float>(0, 3, 10.f));
        ASSERT_EQ(v[5], ElemItem<float>(2, 3, 12.f));
	}
	// slice
	m1.SetSlice(1,2);
	m1.GetSparseData(v);
	if (m1.IsRowMajor())
	{
		ASSERT_EQ(v.size(), 4);
        ASSERT_EQ(v[0], ElemItem<float>(0, 0, 2.f));
        ASSERT_EQ(v[1], ElemItem<float>(0, 2, 8.f));
        ASSERT_EQ(v[2], ElemItem<float>(1, 1, 6.f));
        ASSERT_EQ(v[3], ElemItem<float>(1, 3, 12.f));
	}
	else
	{
		ASSERT_EQ(v.size(), 3);
        ASSERT_EQ(v[0], ElemItem<float>(0, 0, 4.f));
        ASSERT_EQ(v[1], ElemItem<float>(2, 0, 6.f));
        ASSERT_EQ(v[2], ElemItem<float>(1, 1, 8.f));
	}
	// assign whole matrix
	BaseMatrix<float> m2(mft);
	m2.Resize(m1.GetNumRows(), m1.GetNumCols());
	m2.PutSparseData(v);
	ASSERT_TRUE(m2.IsEqualTo(m1));
}

TEST_F(BaseMatrixTests, GetPutSparseData)
{
	TestGetPutSparseData(matrixFormatDenseCol);
	TestGetPutSparseData(matrixFormatDenseRow);
	TestGetPutSparseData(matrixFormatSparseCSC);
	TestGetPutSparseData(matrixFormatSparseCSR);
	TestGetPutSparseData(matrixFormatSparseBSC);
	TestGetPutSparseData(matrixFormatSparseBSR);
}

static void TestTransposeTo(MatrixFormat mft)
{
	BaseMatrix<float> m1(mft);
	std::array<float, 12> array = { 0, 2, 0, 4, 5, 6, 0, 8, 9, 0, 0, 0 };
	m1.Assign(3, 4, array.data(), matrixFlagNone);

	cout << "\t\tmatrix " << m1.FormatStr() << endl;

	BaseMatrix<float> m2; m1.TransposeTo(m2);
	ASSERT_EQ(m2.GetNumRows(), m1.GetNumCols());
	ASSERT_EQ(m2.GetNumCols(), m1.GetNumRows());
	for (size_t j=0; j<m1.GetNumCols(); ++j)
	for (size_t i=0; i<m1.GetNumRows(); ++i)
		ASSERT_EQ(m1.GetItem(i,j), m2.GetItem(j,i));

	BaseMatrix<float> m3;
	m1.Resize(100,1000); Random(m1,5000);
	m1.TransposeTo(m2).TransposeTo(m3);
	ASSERT_TRUE(m3.IsEqualTo(m1));
}

TEST_F(BaseMatrixTests, TransposeTo)
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

	cout << "\t\tmatrix " << m1.FormatStr() << endl;

	size_t rows = 4;
	size_t cols = 3;
	BaseMatrix<float> m2;
	m2.Assign(m1); m2.Reshape(rows,cols);
	if (mft & matrixFormatRowMajor)
	{
		for (size_t j=0; j<m1.GetNumCols(); ++j)
		for (size_t i=0; i<m1.GetNumRows(); ++i)
		{
			size_t n = i*m1.GetNumCols() + j;
			ASSERT_EQ(m1.GetItem(i,j), m2.GetItem(n/cols,n%cols));
		}
	}
	else
	{
		for (size_t j=0; j<m1.GetNumCols(); ++j)
		for (size_t i=0; i<m1.GetNumRows(); ++i)
		{
			size_t n = j*m1.GetNumRows() + i;
			ASSERT_EQ(m1.GetItem(i,j), m2.GetItem(n%rows,n/rows));
		}
	}
}

TEST_F(BaseMatrixTests, Reshape)
{
	TestReshape(matrixFormatDenseCol);
	TestReshape(matrixFormatDenseRow);
	TestReshape(matrixFormatSparseCSC);
	TestReshape(matrixFormatSparseCSR);
	TestReshape(matrixFormatSparseBSC);
	TestReshape(matrixFormatSparseBSR);
}

} } } }
