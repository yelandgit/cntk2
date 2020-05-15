//
// Copyright (c) Microsoft. All rights reserved.
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
//#include "Math/Matrix.h"
//#include "Math/CPUMatrix.h"
//#include "Math/Helpers.h"
#include "gtest/gtest.h"

//#define IDX2C(i, j, ld) (((j) * (ld)) + (i)) // 0 based indexing
//
//#define SIGNUM(v) ((v) > 0.0f ? 1.0f : -1.0f)
//#define SIGNUMZ(v) ((v) == 0.0f ? 0.0f : (SIGNUM(v)))

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

class MatrixTests : public ::testing::Test
{
public:
	static void SetUpTestCase() {}
	static void TearDownTestCase() {}
};

///TEST_F(MatrixTests, Constructors)
///{
///	SingleMatrix a0(c_deviceIdZero);
///	SingleMatrix a1(c_deviceIdZero);
///	SingleMatrix a2(CPUDEVICE);
///	SingleMatrix a3(13, 12, c_deviceIdZero);

///	ASSERT_EQ(0, a0.GetNumRows());
///	ASSERT_EQ(0, a0.GetNumCols());
///	ASSERT_EQ(0, a1.GetNumRows());
///	ASSERT_EQ(0, a1.GetNumCols());
///	ASSERT_EQ(13, a3.GetNumRows());
///	ASSERT_EQ(12, a3.GetNumCols());

///	ASSERT_EQ(a0.GetDeviceId(), c_deviceIdZero);
///	ASSERT_EQ(a1.GetDeviceId(), c_deviceIdZero);
///	ASSERT_EQ(a2.GetDeviceId(), CPUDEVICE);
///	ASSERT_EQ(a3.GetDeviceId(), c_deviceIdZero);
///}

///TEST_F(MatrixTests, MoveTest1)
///{
///	// no moves required
///	SingleMatrix a(c_deviceIdZero);
///	SingleMatrix b(c_deviceIdZero);
///	b.Resize(50, 100);

///	ASSERT_EQ(b.GetNumRows(), 50);
///	ASSERT_EQ(b.GetNumCols(), 100);
///	ASSERT_EQ(a.GetDeviceId(), c_deviceIdZero);
///	ASSERT_EQ(b.GetDeviceId(), c_deviceIdZero);

///	std::swap(a, b);
///	ASSERT_EQ(a.GetNumRows(), 50);
///	ASSERT_EQ(a.GetNumCols(), 100);
///	ASSERT_EQ(a.GetDeviceId(), c_deviceIdZero);
///	ASSERT_EQ(b.GetDeviceId(), c_deviceIdZero);
///}

///TEST_F(MatrixTests, MoveTest2)
///{
///	// potentially a move is required
///	SingleMatrix a(c_deviceIdZero);
///	SingleMatrix b(c_deviceIdZero);
///	b.Resize(50, 100);
///	ASSERT_EQ(b.GetNumRows(), 50);
///	ASSERT_EQ(b.GetNumCols(), 100);
///	ASSERT_EQ(a.GetDeviceId(), c_deviceIdZero);
///	ASSERT_EQ(b.GetDeviceId(), c_deviceIdZero);

///	b(12, 13) = 14; // this will move whole matrix B from GPU to CPU
///	ASSERT_EQ(b.GetDeviceId(), -1);

///	std::swap(a, b); // this will not only swap A and B but will put them to their preferred device (GPU if present)
///	ASSERT_EQ(a.GetNumRows(), 50);
///	ASSERT_EQ(a.GetNumCols(), 100);
///	ASSERT_EQ(b.GetNumRows(), 0);
///	ASSERT_EQ(b.GetNumCols(), 0);
///	ASSERT_EQ(a.GetDeviceId(), -1);
///	ASSERT_EQ(b.GetDeviceId(), c_deviceIdZero);
///}

///TEST_F(MatrixTests, DeepCopy)
///{
///	// This is deep copy, not move
///	SingleMatrix a(c_deviceIdZero);
///	SingleMatrix b(c_deviceIdZero);

///	b.Resize(50, 100);
///	ASSERT_EQ(a.GetNumRows(), 0);
///	ASSERT_EQ(a.GetNumCols(), 0);
///	ASSERT_EQ(b.GetNumRows(), 50);
///	ASSERT_EQ(b.GetNumCols(), 100);

///	b.SetValue(a);
///	ASSERT_EQ(a.GetNumRows(), 0);
///	ASSERT_EQ(a.GetNumCols(), 0);
///	ASSERT_EQ(b.GetNumRows(), 0);
///	ASSERT_EQ(b.GetNumCols(), 0);

///	b.Resize(50, 100);
///	ASSERT_EQ(a.GetNumRows(), 0);
///	ASSERT_EQ(a.GetNumCols(), 0);
///	ASSERT_EQ(b.GetNumRows(), 50);
///	ASSERT_EQ(b.GetNumCols(), 100);

///	b(2, 3) = 9;
///	ASSERT_EQ(b(2, 3), 9);

///	b.SetValue(a);
///	ASSERT_EQ(a.GetNumRows(), 0);
///	ASSERT_EQ(a.GetNumCols(), 0);
///	ASSERT_EQ(b.GetNumRows(), 0);
///	ASSERT_EQ(b.GetNumCols(), 0);
///}

///TEST_F(MatrixTests, InitZero)
///{
///	SingleMatrix a = SingleMatrix::Zeros(12, 32, c_deviceIdZero);
///	ASSERT_EQ(a.GetNumRows(), 12);
///	ASSERT_EQ(a.GetNumCols(), 32);
///	foreach_coord (i, j, a)
///	{
///		ASSERT_EQ(a(i,j), 0.0);
///	}
///}

///TEST_F(MatrixTests, InitEye)
///{
///	SingleMatrix a = SingleMatrix::Eye(56, c_deviceIdZero);
///	ASSERT_EQ(a.GetNumRows(), 56);
///	ASSERT_EQ(a.GetNumCols(), 56);

///	foreach_coord (i, j, a)
///	{
///		if (i != j)
///		{
///			ASSERT_EQ(a(i,j), 0.0);
///		}
///		else
///		{
///			ASSERT_EQ(a(i,j), 1.0);
///		}
///	}
///}

///TEST_F(MatrixTests, InitOnes)
///{
///	SingleMatrix a = SingleMatrix::Ones(12, 56, c_deviceIdZero);
///	ASSERT_EQ(a.GetNumRows(), 12);
///	ASSERT_EQ(a.GetNumCols(), 56);
///	foreach_coord (i, j, a)
///	{
///		ASSERT_EQ(a(i,j), 1.0);
///	}
///}

///TEST_F(MatrixTests, InitGaussianRand)
///{
///	RandomSeedFixture rsf;
///	SingleMatrix a = SingleMatrix::RandomGaussian(640, 230, c_deviceIdZero, 0.0f, 2.0f, rsf.IncrementCounter());
///	ASSERT_EQ(a.GetNumRows(), 640);
///	ASSERT_EQ(a.GetNumCols(), 230);

///	float avg = 0;
///	foreach_coord (i, j, a)
///	{
///		avg += a(i,j);
///	}
///	avg /= (640 * 230);

///	float std = 0;
///	foreach_coord (i, j, a)
///	{
///		std += ((a(i,j) - avg) * (a(i,j) - avg));
///	}
///	std = sqrt(std / (640 * 230));

///	ASSERT_LE(fabs(avg), c_epsilonFloatE1);
///	ASSERT_LE(fabs(std - 2), c_epsilonFloatE1);
///}

///TEST_F(MatrixTests, InitRandomUniform)
///{
///	const float low = -26.3f;
///	const float high = 30.2f;
///	RandomSeedFixture rsf;
///	SingleMatrix a = SingleMatrix::RandomUniform(435, 100, c_deviceIdZero, low, high, rsf.IncrementCounter());
///	bool has_small = false;
///	bool has_big = false;
///	foreach_coord (i, j, a)
///	{
///		ASSERT_GE(a(i,j), low);
///		ASSERT_LE(a(i,j), high);
///		if (a(i,j) < -3)
///		{
///			has_small = true;
///		}
///		if (a(i,j) > 3)
///		{
///			has_big = true;
///		}
///	}
///	ASSERT_TRUE(has_small);
///	ASSERT_TRUE(has_big);
///}

///TEST_F(MatrixTests, SetValueMethodsWithDoubleInstantiation)
///{
///	//Test on ElemType = double
///	// void SetValue(const ElemType v);
///	DoubleMatrix a(32, 12, c_deviceIdZero);
///	ASSERT_EQ(32, a.GetNumRows());
///	ASSERT_EQ(12, a.GetNumCols());
///	ASSERT_EQ(12 * 32, a.GetNumElements());
///	const double v = -32.3451f;
///	a.SetValue(v);
///	foreach_coord(i, j, a)
///	{
///		ASSERT_EQ(v, a(i,j));
///	}

///	// void SetValue(const Matrix<ElemType>& deepCopyFrom);
///	DoubleMatrix b(c_deviceIdZero);
///	b.SetValue(a);
///	foreach_coord(i, j, b)
///	{
///		ASSERT_EQ(v, b(i,j));
///	}

///	// void SetValue(const size_t numRows, const size_t numCols, ElemType *pArray, const bool srcIsColMajor);
///	std::array<double, 7> arrVector = { 123.0f, 0.23f, -22.0f, 63.0f, 43.42f, 324.3f, 99912.0f };

///	double *arr = arrVector.data();
///	b.SetValue(2, 3, b.GetDeviceId(), arr, matrixFlagNone);

///	DoubleMatrix b1(c_deviceIdZero);
///	b1.SetValue(2, 3, b.GetDeviceId(), arr);
///	foreach_coord(i, j, b1)
///	{
///		ASSERT_EQ(arr[IDX2C(i, j, 2)], b(i,j));
///		ASSERT_EQ(arr[IDX2C(i, j, 2)], b1(i,j));
///	}

///	DoubleMatrix bbbb = DoubleMatrix::Zeros(6, 8, c_deviceIdZero);
///	bbbb.SetColumn(arr, 3);
///	for (int i = 0; i < 6; ++i)
///	{
///		ASSERT_EQ(arr[i], bbbb(i, 3));
///	}

///	// void SetDiagonalValue(const ElemType v);
///	DoubleMatrix c(4, 4, c_deviceIdZero);
///	const double val = -0.00332f;
///	c.SetDiagonalValue(val);
///	foreach_coord(i, j, c)
///	{
///		if (i == j)
///			ASSERT_EQ(val, c(i,j));
///		else
///			ASSERT_EQ(0, c(i,j));
///	}

///	// void SetDiagonalValue(const Matrix<ElemType>& vector);
///	DoubleMatrix d(4, 1, c_deviceIdZero);
///	const double val1 = 43.324f;
///	d.SetValue(val1);
///	c.SetDiagonalValue(d);
///	foreach_coord(i, j, c)
///	{
///		if (i == j)
///			ASSERT_EQ(val1, c(i,j));
///		else
///			ASSERT_EQ(0, c(i,j));
///	}

///	// void SetDiagonalValue(const ElemType v);//on non-squared matrix row < col
///	DoubleMatrix c_ns1(4, 6, c_deviceIdZero);
///	const double val_ns1 = -0.00332f;
///	c_ns1.SetValue(0.0f);
///	c_ns1.SetDiagonalValue(val_ns1);
///	foreach_coord(i, j, c_ns1)
///	{
///		if (i == j && i <= 4)
///			ASSERT_EQ(val_ns1, c_ns1(i,j));
///		else
///			ASSERT_EQ(0, c_ns1(i,j));
///	}

///	// void SetDiagonalValue(const Matrix<ElemType>& vector);
///	const double val1_ns1 = 43.324f;
///	c_ns1.SetValue(0.0f);
///	d.SetValue(val1_ns1);
///	c_ns1.SetDiagonalValue(d);
///	foreach_coord(i, j, c_ns1)
///	{
///		if (i == j && i <= 4)
///			ASSERT_EQ(val1, c_ns1(i,j));
///		else
///			ASSERT_EQ(0, c_ns1(i,j));
///	}

///	// void SetDiagonalValue(const ElemType v);//on non-squared matrix row > col
///	DoubleMatrix c_ns2(7, 4, c_deviceIdZero);
///	const double val_ns2 = -0.00332f;
///	c_ns2.SetValue(0.0f);
///	c_ns2.SetDiagonalValue(val_ns2);
///	foreach_coord(i, j, c_ns2)
///	{
///		if (i == j && i <= 4)
///			ASSERT_EQ(val_ns2, c_ns2(i,j));
///		else
///			ASSERT_EQ(0, c_ns2(i,j));
///	}

///	// void SetDiagonalValue(const Matrix<ElemType>& vector);
///	DoubleMatrix c_ns2_1(7, 4, c_deviceIdZero);
///	DoubleMatrix dd(4, 1, c_deviceIdZero);
///	const double val1_ns2 = 43.324f;
///	dd.SetValue(val1_ns2);
///	c_ns2_1.SetValue(0.0f);
///	c_ns2_1.SetDiagonalValue(dd);
///	foreach_coord(i, j, c_ns2)
///	{
///		if (i == j && i <= 4)
///			ASSERT_EQ(val1_ns2, c_ns2_1(i,j));
///		else
///			ASSERT_EQ(0, c_ns2_1(i,j));
///	}


///	DoubleMatrix c1(5, 5, c_deviceIdZero);
///	DoubleMatrix d1(1, 5, c_deviceIdZero);
///	double val2 = 0.53f;
///	c1.SetValue(0.0f);
///	d1 = d1.Transpose();
///	d1.SetValue(val2);
///	c1.SetDiagonalValue(d1);
///	foreach_coord(i, j, c1)
///	{
///		if (i == j)
///			ASSERT_EQ(val2, c1(i,j));
///		else
///			ASSERT_EQ(0, c1(i,j));
///	}
///}

///TEST_F(MatrixTests, SetValueMethods)
///{
///	//Test on ElemType = float
///	// void SetValue(const ElemType v);
///	SingleMatrix a(32, 12, c_deviceIdZero);
///	ASSERT_EQ(32, a.GetNumRows());
///	ASSERT_EQ(12, a.GetNumCols());
///	ASSERT_EQ(12 * 32, a.GetNumElements());
///	const float v = -32.3451f;
///	a.SetValue(v);
///	foreach_coord (i, j, a)
///	{
///		ASSERT_EQ(v, a(i,j));
///	}

///	// void SetValue(const Matrix<ElemType>& deepCopyFrom);
///	SingleMatrix b(c_deviceIdZero);
///	b.SetValue(a);
///	foreach_coord (i, j, b)
///	{
///		ASSERT_EQ(v, b(i,j));
///	}

///	// void SetValue(const size_t numRows, const size_t numCols, ElemType *pArray, const bool srcIsColMajor);
///	std::array<float, 7> arrVector = { 123.0f, 0.23f, -22.0f, 63.0f, 43.42f, 324.3f, 99912.0f };

///	float *arr = arrVector.data();
///	b.SetValue(2, 3, b.GetDeviceId(), arr, matrixFlagNone);

///	SingleMatrix b1(c_deviceIdZero);
///	b1.SetValue(2, 3, b.GetDeviceId(), arr);
///	foreach_coord (i, j, b1)
///	{
///		ASSERT_EQ(arr[IDX2C(i, j, 2)], b(i,j));
///		ASSERT_EQ(arr[IDX2C(i, j, 2)], b1(i,j));
///	}

///	SingleMatrix bbbb = SingleMatrix::Zeros(6, 8, c_deviceIdZero);
///	bbbb.SetColumn(arr, 3);
///	for (int i = 0; i < 6; ++i)
///	{
///		ASSERT_EQ(arr[i], bbbb(i, 3));
///	}

///	// void SetDiagonalValue(const ElemType v);
///	SingleMatrix c(4, 4, c_deviceIdZero);
///	const float val = -0.00332f;
///	c.SetDiagonalValue(val);
///	foreach_coord (i, j, c)
///	{
///		if (i == j)
///			ASSERT_EQ(val, c(i,j));
///		else
///			ASSERT_EQ(0, c(i,j));
///	}

///	// void SetDiagonalValue(const Matrix<ElemType>& vector);
///	SingleMatrix d(4, 1, c_deviceIdZero);
///	const float val1 = 43.324f;
///	d.SetValue(val1);
///	c.SetDiagonalValue(d);
///	foreach_coord (i, j, c)
///	{
///		if (i == j)
///			ASSERT_EQ(val1, c(i,j));
///		else
///			ASSERT_EQ(0, c(i,j));
///	}

///	// void SetDiagonalValue(const ElemType v);//on non-squared matrix row < col
///	SingleMatrix c_ns1(4, 6, c_deviceIdZero);
///	const float val_ns1 = -0.00332f;
///	c_ns1.SetValue(0.0f);
///	c_ns1.SetDiagonalValue(val_ns1);
///	foreach_coord(i, j, c_ns1)
///	{
///		if (i == j && i <= 4)
///			ASSERT_EQ(val_ns1, c_ns1(i,j));
///		else
///			ASSERT_EQ(0, c_ns1(i,j));
///	}

///	// void SetDiagonalValue(const Matrix<ElemType>& vector);
///	const float val1_ns1 = 43.324f;
///	c_ns1.SetValue(0.0f);
///	d.SetValue(val1_ns1);
///	c_ns1.SetDiagonalValue(d);
///	foreach_coord(i, j, c_ns1)
///	{
///		if (i == j && i <= 4)
///			ASSERT_EQ(val1, c_ns1(i,j));
///		else
///			ASSERT_EQ(0, c_ns1(i,j));
///	}

///	// void SetDiagonalValue(const ElemType v);//on non-squared matrix row > col
///	SingleMatrix c_ns2(7, 4, c_deviceIdZero);
///	const float val_ns2 = -0.00332f;
///	c_ns2.SetValue(0.0f);
///	c_ns2.SetDiagonalValue(val_ns2);
///	foreach_coord(i, j, c_ns2)
///	{
///		if (i == j && i <= 4)
///			ASSERT_EQ(val_ns2, c_ns2(i,j));
///		else
///			ASSERT_EQ(0, c_ns2(i,j));
///	}

///	// void SetDiagonalValue(const Matrix<ElemType>& vector);
///	SingleMatrix c_ns2_1(7, 4, c_deviceIdZero);
///	SingleMatrix dd(4, 1, c_deviceIdZero);
///	const float val1_ns2 = 43.324f;
///	dd.SetValue(val1_ns2);
///	c_ns2_1.SetValue(0.0f);
///	c_ns2_1.SetDiagonalValue(dd);
///	foreach_coord(i, j, c_ns2)
///	{
///		if (i == j && i <= 4)
///			ASSERT_EQ(val1_ns2, c_ns2_1(i,j));
///		else
///			ASSERT_EQ(0, c_ns2_1(i,j));
///	}


///	SingleMatrix c1(5, 5, c_deviceIdZero);
///	SingleMatrix d1(1, 5, c_deviceIdZero);
///	float val2 = 0.53f;
///	c1.SetValue(0.0f);
///	d1 = d1.Transpose();
///	d1.SetValue(val2);
///	c1.SetDiagonalValue(d1);
///	foreach_coord (i, j, c1)
///	{
///		if (i == j)
///			ASSERT_EQ(val2, c1(i,j));
///		else
///			ASSERT_EQ(0, c1(i,j));
///	}
///}

///TEST_F(MatrixTests, TransposeTest)
///{
///	RandomSeedFixture rsf;
///	SingleMatrix a = SingleMatrix::RandomGaussian(64, 23, c_deviceIdZero, 0, 2, rsf.IncrementCounter());
///	ASSERT_EQ(64, a.GetNumRows());
///	ASSERT_EQ(23, a.GetNumCols());

///	SingleMatrix b = a.Transpose();

///	ASSERT_EQ(23, b.GetNumRows());
///	ASSERT_EQ(64, b.GetNumCols());

///	foreach_coord (i, j, a)
///	{
///		ASSERT_EQ(a(i,j), b(j, i));
///	}
///}

///TEST_F(MatrixTests, MultiAndDiv)
///{
///	SingleMatrix m0(2, 3, c_deviceIdZero);
///	m0(0, 0) = 1;
///	m0(0, 1) = 2;
///	m0(0, 2) = 3;
///	m0(1, 0) = 4;
///	m0(1, 1) = 5;
///	m0(1, 2) = 6;

///	SingleMatrix m00(2, 3, c_deviceIdZero);
///	m00(0, 0) = 10;
///	m00(0, 1) = 20;
///	m00(0, 2) = 30;
///	m00(1, 0) = 40;
///	m00(1, 1) = 50;
///	m00(1, 2) = 60;

///	SingleMatrix m1(2, 3, c_deviceIdZero);
///	m1.Reshape(3, 2);
///	m1(0, 0) = 11;
///	m1(0, 1) = 15;
///	m1(1, 0) = 14;
///	m1(1, 1) = 13;
///	m1(2, 0) = 12;
///	m1(2, 1) = 16;

///	SingleMatrix m2(2, 2, c_deviceIdZero);
///	m2(0, 0) = 75;
///	m2(0, 1) = 89;
///	m2(1, 0) = 186;
///	m2(1, 1) = 221;

///	SingleMatrix m3 = m0 * m1;
///	ASSERT_TRUE(m3.IsEqualTo(m2));

///	m3 = m0 * 10;
///	ASSERT_TRUE(m3.IsEqualTo(m00));

///	m3 = m3 / 10;
///	ASSERT_TRUE(m3.IsEqualTo(m0));

///	m3 *= 10;
///	ASSERT_TRUE(m3.IsEqualTo(m00));

///	m3 /= 10;
///	ASSERT_TRUE(m3.IsEqualTo(m0));

///	SingleMatrix::MultiplyAndWeightedAdd(1, m0, false, m1, false, 0, m3);
///	ASSERT_TRUE(m3.IsEqualTo(m2));

///	m1.Reshape(2, 3);
///	SingleMatrix::MultiplyAndWeightedAdd(1, m0, false, m1, true, 0, m3);
///	m2(0, 0) = 74;
///	m2(0, 1) = 92;
///	m2(1, 0) = 182;
///	m2(1, 1) = 227;
///	ASSERT_TRUE(m3.IsEqualTo(m2));

///	SingleMatrix::MultiplyAndWeightedAdd(10, m0, false, m1, true, 2, m3);
///	m2(0, 0) = 888;
///	m2(0, 1) = 1104;
///	m2(1, 0) = 2184;
///	m2(1, 1) = 2724;
///	ASSERT_TRUE(m3.IsEqualTo(m2));

///	SingleMatrix::MultiplyAndWeightedAdd(1, m0, true, m1, false, 0, m3);
///	m2.Resize(3, 3);
///	m2(0, 0) = 67;
///	m2(0, 1) = 72;
///	m2(0, 2) = 77;
///	m2(1, 0) = 92;
///	m2(1, 1) = 99;
///	m2(1, 2) = 106;
///	m2(2, 0) = 117;
///	m2(2, 1) = 126;
///	m2(2, 2) = 135;
///	ASSERT_TRUE(m3.IsEqualTo(m2));

///	// Multiplications of arbitrary matrix with 1x1 matrix

///	SingleMatrix a(2, 3, c_deviceIdZero);
///	a(0, 0) = 1;
///	a(0, 1) = 2;
///	a(0, 2) = 3;
///	a(1, 0) = 4;
///	a(1, 1) = 5;
///	a(1, 2) = 6;

///	SingleMatrix b = SingleMatrix::Eye(1, c_deviceIdZero);

///	SingleMatrix c = a * b;
///	ASSERT_TRUE(c.IsEqualTo(a));
///	c = b * a;
///	ASSERT_TRUE(c.IsEqualTo(a));
///	b(0, 0) = 0.5;
///	b.InplaceAbs();
///	c = a * b;

///	SingleMatrix d(2, 3, c_deviceIdZero);
///	d(0, 0) = 0.5;
///	d(0, 1) = 1;
///	d(0, 2) = 1.5;
///	d(1, 0) = 2;
///	d(1, 1) = 2.5;
///	d(1, 2) = 3;
///	ASSERT_TRUE(c.IsEqualTo(d));
///}

///TEST_F(MatrixTests, Transpose)
///{
///	SingleMatrix m0(2, 3, c_deviceIdZero);
///	m0(0, 0) = 1;
///	m0(0, 1) = 2;
///	m0(0, 2) = 3;
///	m0(1, 0) = 4;
///	m0(1, 1) = 5;
///	m0(1, 2) = 6;

///	SingleMatrix m1(3, 2, c_deviceIdZero);
///	m1(0, 0) = 1;
///	m1(0, 1) = 4;
///	m1(1, 0) = 2;
///	m1(1, 1) = 5;
///	m1(2, 0) = 3;
///	m1(2, 1) = 6;

///	SingleMatrix m2 = m0.Transpose();
///	ASSERT_TRUE(m2.IsEqualTo(m1, c_epsilonFloatE4));

///	m2.AssignTransposeOf(m1);
///	ASSERT_TRUE(m2.IsEqualTo(m0, c_epsilonFloatE4));
///}

///TEST_F(MatrixTests, AddAndSub)
///{
///	SingleMatrix m0(2, 3, c_deviceIdZero);
///	m0(0, 0) = 1;
///	m0(0, 1) = 2;
///	m0(0, 2) = 3;
///	m0(1, 0) = 4;
///	m0(1, 1) = 5;
///	m0(1, 2) = 6;

///	SingleMatrix m1(2, 3, c_deviceIdZero);
///	m1(0, 0) = 11;
///	m1(0, 1) = 12;
///	m1(0, 2) = 13;
///	m1(1, 0) = 14;
///	m1(1, 1) = 15;
///	m1(1, 2) = 16;

///	SingleMatrix m2(2, 3, c_deviceIdZero);
///	m2(0, 0) = 12;
///	m2(0, 1) = 14;
///	m2(0, 2) = 16;
///	m2(1, 0) = 18;
///	m2(1, 1) = 20;
///	m2(1, 2) = 22;

///	SingleMatrix m3 = m2 - m0;
///	ASSERT_TRUE(m3.IsEqualTo(m1));

///	m3 += m0;
///	ASSERT_TRUE(m3.IsEqualTo(m2));

///	m3 = m0 + 10;
///	ASSERT_TRUE(m3.IsEqualTo(m1));

///	m3 -= 10;
///	ASSERT_TRUE(m3.IsEqualTo(m0));

///	m3 = m1 + m0;
///	ASSERT_TRUE(m3.IsEqualTo(m2));
///	SingleMatrix m4 = SingleMatrix::Eye(3, c_deviceIdZero);

///	m3 -= m0;
///	ASSERT_TRUE(m3.IsEqualTo(m1));

///	m3 = m1 - 10;
///	ASSERT_TRUE(m3.IsEqualTo(m0));

///	SingleMatrix m33(m3.DeepClone());
///	m3 += 10;
///	ASSERT_TRUE(m3.IsEqualTo(m1));

///	SingleMatrix m55 = SingleMatrix::Eye(1, c_deviceIdZero);
///	m55(0, 0) = 10;
///	m55.InplaceAbs();
///	m33 += m55;
///	ASSERT_TRUE(m33.IsEqualTo(m1));
///	m33 -= 10;
///	m33 = m33 + 10;
///	ASSERT_TRUE(m33.IsEqualTo(m1));
///}

///TEST_F(MatrixTests, ElementOps)
///{
///	SingleMatrix m0(2, 3, c_deviceIdZero);
///	m0(0, 0) = 1;
///	m0(0, 1) = 2;
///	m0(0, 2) = 3;
///	m0(1, 0) = 4;
///	m0(1, 1) = 5;
///	m0(1, 2) = 6;

///	SingleMatrix m00(2, 3, c_deviceIdZero);
///	m00(0, 0) = 1.0f;
///	m00(0, 1) = static_cast<float>(1 / 2.0);
///	m00(0, 2) = static_cast<float>(1 / 3.0);
///	m00(1, 0) = static_cast<float>(1 / 4.0);
///	m00(1, 1) = static_cast<float>(1 / 5.0);
///	m00(1, 2) = static_cast<float>(1 / 6.0);

///	SingleMatrix m1(2, 3, c_deviceIdZero);
///	m1(0, 0) = 1;
///	m1(0, 1) = 1;
///	m1(0, 2) = 1;
///	m1(1, 0) = 1;
///	m1(1, 1) = 1;
///	m1(1, 2) = 1;

///	SingleMatrix m3(c_deviceIdZero);
///	m3.AssignElementProductOf(m0, m00);
///	ASSERT_TRUE(m3.IsEqualTo(m1, c_epsilonFloatE4));

///	SingleMatrix m4 = SingleMatrix::Zeros(2, 3, c_deviceIdZero);
///	m4.SetValue(m4.AddElementProductOf(m0, m00));
///	ASSERT_TRUE(m4.IsEqualTo(m1, c_epsilonFloatE4));

///	m3 = m0 ^ 4;
///	SingleMatrix m2(2, 3, c_deviceIdZero);
///	m2(0, 0) = 1;
///	m2(0, 1) = 16;
///	m2(0, 2) = 81;
///	m2(1, 0) = 256;
///	m2(1, 1) = 625;
///	m2(1, 2) = 1296;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE3));

///	m3.SetValue(m0);
///	m3 ^= 4;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE3));

///	m3.SetValue(m0);
///	m3.ElementMultiplyWith(m00);
///	ASSERT_TRUE(m3.IsEqualTo(m1, c_epsilonFloatE3));

///	m3.SetValue(m0);
///	m3.ElementInverse();
///	ASSERT_TRUE(m3.IsEqualTo(m00, c_epsilonFloatE3));

///	m2(0, 0) = 0.7311f;
///	m2(0, 1) = 0.8808f;
///	m2(0, 2) = 0.9526f;
///	m2(1, 0) = 0.9820f;
///	m2(1, 1) = 0.9933f;
///	m2(1, 2) = 0.9975f;
///	m3.AssignElementDivisionOf(m2, m0);
///	m2.ElementMultiplyWith(m00);
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));

///	m3.SetValue(m0);
///	m3.InplaceSigmoid();
///	m2(0, 0) = 0.7311f;
///	m2(0, 1) = 0.8808f;
///	m2(0, 2) = 0.9526f;
///	m2(1, 0) = 0.9820f;
///	m2(1, 1) = 0.9933f;
///	m2(1, 2) = 0.9975f;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));

///	m3.SetValue(m0);
///	m3.InplaceTanh();
///	m2(0, 0) = 0.7616f;
///	m2(0, 1) = 0.9640f;
///	m2(0, 2) = 0.9951f;
///	m2(1, 0) = 0.9993f;
///	m2(1, 1) = 0.9999f;
///	m2(1, 2) = 1.0000f;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));

///	m3.SetValue(m0);
///	m3.InplaceLogSoftmax(true);
///	m3.InplaceExp();
///	m2(0, 0) = 0.0474f;
///	m2(0, 1) = 0.0474f;
///	m2(0, 2) = 0.0474f;
///	m2(1, 0) = 0.9526f;
///	m2(1, 1) = 0.9526f;
///	m2(1, 2) = 0.9526f;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));

///	m3.SetValue(m0);
///	m3.InplaceLogSoftmax(false);
///	m3.InplaceExp();
///	m2(0, 0) = 0.0900f;
///	m2(0, 1) = 0.2447f;
///	m2(0, 2) = 0.6652f;
///	m2(1, 0) = 0.0900f;
///	m2(1, 1) = 0.2447f;
///	m2(1, 2) = 0.6652f;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));

///	m3.SetValue(m0);
///	m3.InplaceHardmax(true);
///	m2(0, 0) = 0.0f;
///	m2(0, 1) = 0.0f;
///	m2(0, 2) = 0.0f;
///	m2(1, 0) = 1.0f;
///	m2(1, 1) = 1.0f;
///	m2(1, 2) = 1.0f;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));

///	m3.SetValue(m0);
///	m3.InplaceSqrt();
///	m2(0, 0) = 1.0f;
///	m2(0, 1) = 1.4142f;
///	m2(0, 2) = 1.7321f;
///	m2(1, 0) = 2.0f;
///	m2(1, 1) = 2.2361f;
///	m2(1, 2) = 2.4495f;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));

///	m3.SetValue(m0);
///	m3.InplaceExp();
///	m2(0, 0) = 2.7183f;
///	m2(0, 1) = 7.3891f;
///	m2(0, 2) = 20.0855f;
///	m2(1, 0) = 54.5982f;
///	m2(1, 1) = 148.4132f;
///	m2(1, 2) = 403.4288f;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));

///	m3.SetValue(m0);
///	m3.InplaceExp();
///	m2(0, 0) = 2.7183f;
///	m2(0, 1) = 7.3891f;
///	m2(0, 2) = 20.0855f;
///	m2(1, 0) = 54.5982f;
///	m2(1, 1) = 148.4132f;
///	m2(1, 2) = 403.4288f;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));

///	m3.InplaceLog();
///	ASSERT_TRUE(m3.IsEqualTo(m0, c_epsilonFloatE4));

///	m3.SetValue(m0);
///	m3.InplaceTruncateBottom(2);
///	m2(0, 0) = 2;
///	m2(0, 1) = 2;
///	m2(0, 2) = 3;
///	m2(1, 0) = 4;
///	m2(1, 1) = 5;
///	m2(1, 2) = 6;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE3));

///	m3.SetValue(m0);
///	m3.InplaceTruncateTop(4);
///	m2(0, 0) = 1;
///	m2(0, 1) = 2;
///	m2(0, 2) = 3;
///	m2(1, 0) = 4;
///	m2(1, 1) = 4;
///	m2(1, 2) = 4;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE3));
///}

///TEST_F(MatrixTests, ColumnElementMultiply)
///{
///	RandomSeedFixture rsf;
///	CPUMatrix<float> mcpu = CPUMatrix<float>::RandomUniform(429, 1024, -3.4f, 1, rsf.IncrementCounter());
///	CPUMatrix<float> acpu = CPUMatrix<float>::Ones(429, 1);
///	CPUMatrix<float> mcpuCopy(mcpu);

///	mcpu.ColumnElementMultiplyWith(acpu);
///	ASSERT_TRUE(mcpuCopy.IsEqualTo(mcpu, c_epsilonFloatE4));

///	Matrix<float> m = Matrix<float>::RandomUniform(429, 1024, c_deviceIdZero, -3.4f, 1, rsf.IncrementCounter());
///	Matrix<float> a = Matrix<float>::Ones(429, 1, c_deviceIdZero);
///	Matrix<float> mCopy(m.DeepClone());

///	m.ColumnElementMultiplyWith(a);
///	ASSERT_TRUE(mCopy.IsEqualTo(m, c_epsilonFloatE4));

///	CPUMatrix<float> mc1 = CPUMatrix<float>::RandomUniform(429, 1024, -3.4f, 1, rsf.IncrementCounter());
///	CPUMatrix<float> mc2 = CPUMatrix<float>::RandomUniform(429, 1, 0, 3, rsf.IncrementCounter());
///	mc1.ColumnElementMultiplyWith(mc2);

///	Matrix<float> m1(mc1.GetNumRows(), mc1.GetNumCols(), mc1.GetBuffer(), c_deviceIdZero, matrixFlagNone);
///	Matrix<float> m2(mc2.GetNumRows(), mc2.GetNumCols(), mc2.GetBuffer(), c_deviceIdZero, matrixFlagNone);
///	m1.ColumnElementMultiplyWith(m2);
///	foreach_coord (i, j, m2)
///	{
///		ASSERT_LT(fabs(m2(i,j) - mc2(i,j)), c_epsilonFloatE5);
///	}
///}

///TEST_F(MatrixTests, AssignXOf)
///{
///	// AssignDifferenceOf
///	RandomSeedFixture rsf;
///	Matrix<float> a = Matrix<float>::RandomUniform(429, 1024, c_deviceIdZero, 5, 32, rsf.IncrementCounter());
///	Matrix<float> b = Matrix<float>::RandomUniform(429, 1024, c_deviceIdZero, 5, 32, rsf.IncrementCounter());
///	Matrix<float> c(c_deviceIdZero);

///	c.AssignDifferenceOf(a, b);
///	foreach_coord (i, j, c)
///	{
///		ASSERT_EQ(c(i,j), a(i,j) - b(i,j));
///	}
///	a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

///	float x = 234.2f;
///	c.AssignDifferenceOf(a, x);
///	foreach_coord (i, j, c)
///	{
///		ASSERT_EQ(c(i,j), a(i,j) - x);
///	}
///	a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

///	c.AssignDifferenceOf(x, a);
///	foreach_coord (i, j, c)
///	{
///		ASSERT_EQ(c(i,j), x - a(i,j));
///	}
///	a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

///	c.AssignDifferenceOf(1, a);
///	foreach_coord (i, j, c)
///	{
///		ASSERT_EQ(c(i,j), 1 - a(i,j));
///	}
///	a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

///	// AssignSumOf
///	c.AssignSumOf(a, b);
///	foreach_coord (i, j, c)
///	{
///		ASSERT_EQ(c(i,j), a(i,j) + b(i,j));
///	}
///	a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

///	// Check for self-assignment (c = c + b)
///	auto tolerance = 5e-5;
///	c.AssignSumOf(c, b);
///	foreach_coord (i, j, c)
///	{
///		ASSERT_NEAR(c(i,j), a(i,j) + 2 * b(i,j), tolerance);
///	}
///	a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

///	// Check for self-assignment (c = b + c)
///	c.AssignSumOf(b, c);
///	foreach_coord (i, j, c)
///	{
///		ASSERT_NEAR(c(i,j), a(i,j) + 3 * b(i,j), tolerance);
///	}
///	a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

///	// Check for self-assignment (c = c + a .* c)
///	c.AssignSumOf(a, b);
///	c.AddElementProductOf(a, c);
///	foreach_coord(i, j, c)
///	{
///		float ab = a(i,j) + b(i,j);
///		ASSERT_NEAR(c(i,j), ab + a(i,j)*ab, tolerance);
///	}
///	a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

///	// Check for self-assignment (c = c + c .* a)
///	c.AssignSumOf(a, b);
///	c.AddElementProductOf(c, a);
///	foreach_coord(i, j, c)
///	{
///		float ab = a(i,j) + b(i,j);
///		ASSERT_NEAR(c(i,j), ab + a(i,j)*ab, tolerance);
///	}
///	a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

///	// Check for self-assignment (c = c + c .* c)
///	c.AssignSumOf(a, b);
///	c.AddElementProductOf(c, c);
///	foreach_coord(i, j, c)
///	{
///		float ab = a(i,j) + b(i,j);
///		ASSERT_NEAR(c(i,j), ab + ab*ab, tolerance);
///	}
///	a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

///	// AssignElementProductOf
///	c.AssignElementProductOf(a, b);
///	foreach_coord (i, j, c)
///	{
///		ASSERT_EQ(c(i,j), a(i,j) * b(i,j));
///	}
///	a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

///	// AddElementProductOf
///	Matrix<float> c_copy(c.DeepClone());
///	c.AddElementProductOf(a, b);
///	foreach_coord (i, j, c)
///	{
///		ASSERT_EQ(c(i,j), c_copy(i,j) + a(i,j) * b(i,j));
///	}

///	// AssignSigmoidOf
///	CPUMatrix<float> ac = CPUMatrix<float>::RandomUniform(429, 1024, 5, 32, rsf.IncrementCounter());
///	CPUMatrix<float> bc = CPUMatrix<float>::RandomUniform(429, 1024, -5, 12, rsf.IncrementCounter());
///	Matrix<float> d(ac.GetNumRows(), ac.GetNumCols(), ac.GetBuffer(), c_deviceIdZero, matrixFlagNone);
///	Matrix<float> e(bc.GetNumRows(), bc.GetNumCols(), bc.GetBuffer(), c_deviceIdZero, matrixFlagNone);
///	ac.AssignSigmoidOf(bc);
///	d.AssignSigmoidOf(e);
///	foreach_coord (i, j, ac)
///	{
///		ASSERT_LT(fabs(ac(i,j) - d(i,j)), c_epsilonFloatE5);
///	}

///	// AssignSignOf
///	Matrix<float> m1 = Matrix<float>::RandomUniform(42, 12, c_deviceIdZero, -5, 12, rsf.IncrementCounter());
///	Matrix<float> m2(4, 5, c_deviceIdZero);
///	m2.AssignSignOf(m1);
///	foreach_coord (i, j, m1)
///	{
///		float v = m1(i,j);
///		float expected = SIGNUMZ(v);
///		float actual = m2(i,j);
///		ASSERT_EQ(expected, actual);
///	}

///	Matrix<float> m3 = Matrix<float>::RandomUniform(42, 12, c_deviceIdZero, -5, 2, rsf.IncrementCounter());
///	Matrix<float> m4(m3.DeepClone());
///	m3.AddSignOf(m1);
///	foreach_coord (i, j, m3)
///	{
///		float v = m1(i,j);
///		ASSERT_EQ(m4(i,j) + SIGNUMZ(v), m3(i,j));
///	}

///	// AssignTruncateBottom and Top
///	Matrix<float> m5(2, 2, c_deviceIdZero);
///	m5(0, 0) = 1;
///	m5(0, 1) = 2;
///	m5(1, 0) = 3;
///	m5(1, 1) = 4;

///	Matrix<float> m6(c_deviceIdZero);
///	m6.AssignTruncateBottomOf(m5, 3);
///	ASSERT_EQ(3, m6(0, 0));
///	ASSERT_EQ(3, m6(0, 1));
///	ASSERT_EQ(3, m6(1, 0));
///	ASSERT_EQ(4, m6(1, 1));

///	Matrix<float> m7(c_deviceIdZero);
///	m7.AssignTruncateTopOf(m5, 3);
///	ASSERT_EQ(1, m7(0, 0));
///	ASSERT_EQ(2, m7(0, 1));
///	ASSERT_EQ(3, m7(1, 0));
///	ASSERT_EQ(3, m7(1, 1));
///}

///TEST_F(MatrixTests, SumOfElements)
///{
///	Matrix<float> m = Matrix<float>::Ones(429, 1024, c_deviceIdZero);
///	float sum = m.SumOfElements();
///	ASSERT_EQ(429 * 1024, sum);

///	CPUMatrix<float> mcpu = CPUMatrix<float>::Ones(429, 1024);
///	float sumCPU = mcpu.SumOfElements();
///	ASSERT_EQ(429 * 1024, sumCPU);

///	Matrix<float> m1 = Matrix<float>::Ones(42, 332, c_deviceIdZero);
///	m1 *= -1;
///	float sum1 = m1.SumOfElements();
///	ASSERT_EQ(-1 * 42 * 332, sum1);

///	Matrix<float> m2 = Matrix<float>::Ones(3, 2, c_deviceIdZero);
///	m2 *= -1;
///	float sum2 = m2.SumOfElements();
///	ASSERT_EQ(-1 * 3 * 2, sum2);
///}

///TEST_F(MatrixTests, ColumnSlice)
///{
///	std::array<float, 6> arr = { 1, 2, 3, 4, 5, 6};
///	auto *fArray = arr.data();

///	Matrix<float> m0(2, 3, fArray, c_deviceIdZero, matrixFlagNone);
///	Matrix<float> m1(2, 2, fArray, c_deviceIdZero, matrixFlagNone);
///	Matrix<float> m2 = m0.ColumnSlice(0, 2);
///	ASSERT_TRUE(m2.IsEqualTo(m1, c_epsilonFloatE4));

///	Matrix<float> m3(2, 2, fArray + 2, c_deviceIdZero, matrixFlagNone);
///	m2 = m0.ColumnSlice(1, 2);
///	ASSERT_TRUE(m2.IsEqualTo(m3, c_epsilonFloatE4));

///	RandomSeedFixture rsf;
///	size_t k = 100, n = 20, m = 50;
///	Matrix<float> ag(k, n, c_deviceIdZero);
///	ag.SetUniformRandomValue(-1, 1, rsf.IncrementCounter());

///	Matrix<float> bg(n, m, c_deviceIdZero);
///	bg.SetUniformRandomValue(-1, 1, rsf.IncrementCounter());

///	Matrix<float> cg(k, m, c_deviceIdZero);
///	cg.SetUniformRandomValue(-1, 1, rsf.IncrementCounter());

///	Matrix<float> dg(k, m, c_deviceIdZero);
///	dg.AssignValuesOf(cg);

///	Matrix<float>::MultiplyAndAdd(ag, false, bg, false, dg);
///	for (int i = 0; i < m; i++)
///	{
///		Matrix<float> colBg = bg.ColumnSlice(i, 1);
///		Matrix<float> colCg = cg.ColumnSlice(i, 1);
///		Matrix<float>::MultiplyAndAdd(ag, false, colBg, false, colCg);
///	}
///	ASSERT_TRUE(cg.IsEqualTo(dg, c_epsilonFloatE4));
///}

///TEST_F(MatrixTests, KhatriRaoProduct)
///{
///	std::array<float, 24> arr =
///		{ 0, 0, 0, 0, 0, 0,
///		0, 0, 0, 0, 0, 0,
///		0, 0, 0, 0, 0, 0,
///		0, 0, 0, 0, 0, 0};

///	auto *fArray = arr.data();
///	fArray[0] = 0.8147f;
///	fArray[3] = 0.9134f;
///	fArray[6] = 0.2785f;
///	fArray[9] = 0.9649f;
///	fArray[1] = 0.9058f;
///	fArray[4] = 0.6324f;
///	fArray[7] = 0.5469f;
///	fArray[10] = 0.1576f;
///	fArray[2] = 0.1270f;
///	fArray[5] = 0.0975f;
///	fArray[8] = 0.9575f;
///	fArray[11] = 0.9706f;
///	Matrix<float> a(3, 4, fArray, c_deviceIdZero);

///	fArray[0] = 0.9572f;
///	fArray[2] = 0.8003f;
///	fArray[4] = 0.4218f;
///	fArray[6] = 0.7922f;
///	fArray[1] = 0.4854f;
///	fArray[3] = 0.1419f;
///	fArray[5] = 0.9157f;
///	fArray[7] = 0.9595f;
///	Matrix<float> b(2, 4, fArray, c_deviceIdZero);

///	fArray[0] = 0.7798f;
///	fArray[6] = 0.7310f;
///	fArray[12] = 0.1175f;
///	fArray[18] = 0.7644f;
///	fArray[1] = 0.8670f;
///	fArray[7] = 0.5061f;
///	fArray[13] = 0.2307f;
///	fArray[19] = 0.1249f;
///	fArray[2] = 0.1215f;
///	fArray[8] = 0.0781f;
///	fArray[14] = 0.4038f;
///	fArray[20] = 0.7689f;
///	fArray[3] = 0.3954f;
///	fArray[9] = 0.1296f;
///	fArray[15] = 0.2550f;
///	fArray[21] = 0.9258f;
///	fArray[4] = 0.4396f;
///	fArray[10] = 0.0897f;
///	fArray[16] = 0.5008f;
///	fArray[22] = 0.1512f;
///	fArray[5] = 0.0616f;
///	fArray[11] = 0.0138f;
///	fArray[17] = 0.8768f;
///	fArray[23] = 0.9313f;
///	Matrix<float> d(6, 4, fArray, c_deviceIdZero);

///	Matrix<float> c(c_deviceIdZero);
///	c.AssignKhatriRaoProductOf(a, b);
///	ASSERT_TRUE(c.IsEqualTo(d, c_epsilonFloatE4));
///}

///TEST_F(MatrixTests, AddColumnReshapeProductOf)
///{
///	std::array<float, 12> arr =
///		{ 0, 0, 0, 0, 0, 0,
///		0, 0, 0, 0, 0, 0};

///	auto *fArray = arr.data();
///	fArray[0] = 0.6557f;
///	fArray[6] = 0.7431f;
///	fArray[1] = 0.0357f;
///	fArray[7] = 0.3922f;
///	fArray[2] = 0.8491f;
///	fArray[8] = 0.6555f;
///	fArray[3] = 0.9340f;
///	fArray[9] = 0.1712f;
///	fArray[4] = 0.6787f;
///	fArray[10] = 0.7060f;
///	fArray[5] = 0.7577f;
///	fArray[11] = 0.0318f;
///	Matrix<float> a(6, 2, fArray, c_deviceIdZero);

///	fArray[0] = 0.2769f;
///	fArray[3] = 0.8235f;
///	fArray[1] = 0.0462f;
///	fArray[4] = 0.6948f;
///	fArray[2] = 0.0971f;
///	fArray[5] = 0.3171f;
///	Matrix<float> b(3, 2, fArray, c_deviceIdZero);

///	fArray[0] = 0.2867f;
///	fArray[2] = 1.2913f;
///	fArray[1] = 0.1266f;
///	fArray[3] = 0.4520f;
///	Matrix<float> d0(2, 2, fArray, c_deviceIdZero);

///	fArray[0] = 0.2657f;
///	fArray[2] = 1.0923f;
///	fArray[1] = 0.3636f;
///	fArray[3] = 0.6416f;
///	Matrix<float> d1(2, 2, fArray, c_deviceIdZero);

///	Matrix<float> c(2, 2, c_deviceIdZero);
///	c.SetValue(0.0f);
///	c.AddColumnReshapeProductOf(a, b, false);
///	ASSERT_TRUE(c.IsEqualTo(d0, c_epsilonFloatE4));

///	c.SetValue(0.0f);
///	c.AddColumnReshapeProductOf(a, b, true);
///	ASSERT_TRUE(c.IsEqualTo(d1, c_epsilonFloatE4));
///}

///TEST_F(MatrixTests, Copy)
///{
///	// Matrices are stored as column-major so below is 4x3 matrix.
///	const size_t crow = 4;
///	const size_t ccol = 3;
///	float src[] = {
///		1.0f, 3.0f, 4.0f, 7.0f,
///		6.0f, 2.0f, 5.0f, 8.0f,
///		4.0f, 9.0f, 3.0f, 1.0f };
///	SingleMatrix srcM(crow, ccol, src, c_deviceIdZero, matrixFlagNone);

///	// Test full copy
///	CPUSingleMatrix actM(crow, ccol);
///	srcM.CopySection(actM.GetNumRows(), actM.GetNumCols(), actM.Data(), actM.GetNumRows());
///	ASSERT_TRUE(actM.IsEqualTo(CPUMatrix<float>(actM.GetNumRows(), actM.GetNumCols(), src, matrixFlagNone)));

///	// Test tile copy
///	actM.Resize(crow-1, ccol-1);
///	actM.SetValue(std::numeric_limits<float>::quiet_NaN());
///	srcM.CopySection(actM.GetNumRows(), actM.GetNumCols(), actM.Data(), actM.GetNumRows());

///	vector<float> expected = { 1.0f, 3.0f, 4.0f, 6.0f, 2.0f, 5.0f };
///	ASSERT_TRUE(actM.IsEqualTo(CPUMatrix<float>(actM.GetNumRows(), actM.GetNumCols(), expected.data(), matrixFlagNone)));
///}

///TEST_F(MatrixTests, HasElement)
///{
///	int cpuCount = 0;
///	for (auto deviceId : {CPUDEVICE, c_deviceIdZero})
///	{
///		if (deviceId<0 && ++cpuCount==2) break;

///		const size_t size = 3;
///		float src[size] = { 0.0f, 1.0f, 2.0f };
///		SingleMatrix m1(1, size, src, deviceId, matrixFlagNone);
///		ASSERT_TRUE(SingleMatrix::HasElement(m1, 1.0f));
///		ASSERT_TRUE(!SingleMatrix::HasElement(m1, -1.0f));

///		auto qnan = std::numeric_limits<float>::quiet_NaN();
///		ASSERT_TRUE(!SingleMatrix::HasElement(m1, qnan));
///		auto posInf = std::numeric_limits<float>::infinity();
///		ASSERT_TRUE(!SingleMatrix::HasElement(m1, posInf));

///		m1(0, 1) = qnan;
///		ASSERT_TRUE(SingleMatrix::HasElement(m1, qnan));

///		m1(0, 1) = posInf;
///		ASSERT_TRUE(SingleMatrix::HasElement(m1, posInf));
///	}
///}

///TEST_F(MatrixTests, VectorMax)
///{
///	// Matrices are stored as column-major so below is 3x2 matrix.
///	float src[] = {
///		1.0f, 3.0f, 4.0f,
///		6.0f, 2.0f, 5.0f };

///	float expectedIdx[] = {
///		2.0f, 1.0f,
///		0.0f, 2.0f };

///	float expectedVal[] = {
///		4.0f, 3.0f,
///		6.0f, 5.0f };

///	int cpuCount = 0;
///	for (auto deviceId : {CPUDEVICE, c_deviceIdZero})
///	{
///		if (deviceId<0 && ++cpuCount==2) break;

///		Matrix<float> expIdx(2, 2, expectedIdx, deviceId, matrixFlagNone);
///		Matrix<float> expVal(2, 2, expectedVal, deviceId, matrixFlagNone);

///		Matrix<float> actual(3, 2, src, deviceId, matrixFlagNone);
///		Matrix<float> actualIdx(deviceId);
///		Matrix<float> actualVal(deviceId);

///		auto topK = 2;
///		actual.VectorMax(actualIdx, actualVal, true, topK);
///		ASSERT_TRUE(actualIdx.IsEqualTo(expIdx));
///		ASSERT_TRUE(actualVal.IsEqualTo(expVal));
///	}
///}

///TEST_F(MatrixTests, AssignNumOfDiff)
///{
///	float labels[] = { 1.0f, 2.0f, 3.0f };

///	// Matrices are stored as column-major so below is 2x3 matrix.
///	float topKResults[] = {
///		1.0f, 3.0f,
///		4.0f, 6.0f,
///		2.0f, 3.0f };

///	int cpuCount = 0;
///	for (auto deviceId : {CPUDEVICE, c_deviceIdZero})
///	{
///		if (deviceId<0 && ++cpuCount==2) break;

///		Matrix<float> lbl(1, 3, labels, deviceId, matrixFlagNone);
///		Matrix<float> topKRes(2, 3, topKResults, deviceId, matrixFlagNone);

///		Matrix<float> actual(deviceId);
///		actual.AssignNumOfDiff(lbl, topKRes, true);

///		float expectedDiff = 1.0;
///		ASSERT_EQ(expectedDiff, actual.Get00Element());
///	}
///}

///TEST_F(MatrixTests, Scale)
///{
///	const float low = -1.0f;
///	const float high = 1.0f;
///	float alpha = 0.7713f;
///	int cpuCount = 0;
///	RandomSeedFixture rsf;
///	for (auto deviceId : {CPUDEVICE, c_deviceIdZero})
///	{
///		if (deviceId<0 && ++cpuCount==2) break;

///		auto a1 = SingleMatrix::RandomUniform(7, 11, deviceId, low, high, rsf.IncrementCounter());
///		auto a2 = a1.DeepClone();
///		ASSERT_TRUE(a1.IsEqualTo(a2));

///		auto b1 = SingleMatrix::RandomUniform(7, 11, deviceId, low, high, rsf.IncrementCounter());
///		auto b2 = b1.DeepClone();
///		ASSERT_TRUE(b1.IsEqualTo(b2));

///		Matrix<float>::ScaleAndAdd(alpha, b1, a1);

///		Matrix<float>::Scale(alpha, b2);
///		a2 += b2;

///		// BUGBUG: this test currently fails on GPU.
///		if (deviceId != CPUDEVICE)
///			continue;

///		// TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
///		// ASSERT_TRUE(a1.IsEqualTo(a2));
///		ASSERT_TRUE(a1.IsEqualTo(a2, c_epsilonFloatE5));
///	}
///}

///TEST_F(MatrixTests, SGDUpdate)
///{
///	const float low = -1.0f;
///	const float high = 1.0f;
///	float lr = 0.77f;
///	int cpuCount = 0;
///	RandomSeedFixture rsf;
///	for (auto deviceId : {CPUDEVICE, c_deviceIdZero})
///	{
///		if (deviceId<0 && ++cpuCount==2) break;

///		auto p1 = SingleMatrix::RandomUniform(12, 13, deviceId, low, high, rsf.IncrementCounter());
///		auto p2 = p1.DeepClone();
///		ASSERT_TRUE(p1.IsEqualTo(p2));

///		auto g1 = SingleMatrix::RandomUniform(12, 13, deviceId, low, high, rsf.IncrementCounter());
///		auto g2 = g1.DeepClone();
///		ASSERT_TRUE(g1.IsEqualTo(g2));

///		auto sg1 = SingleMatrix::RandomUniform(12, 13, deviceId, low, high, rsf.IncrementCounter());
///		auto sg2 = sg1.DeepClone();
///		ASSERT_TRUE(sg1.IsEqualTo(sg2));

///		for (; lr > 0.01; lr = lr / 2)
///		{
///			if (deviceId != CPUDEVICE)
///			{
///				// g1 is modified inside the GPU version of SGDUpdate, restore the original value here.
///				g1.SetValue(g2);
///			}

///			p1.SGDUpdate(g1, lr);
///			p2.MomentumSGDUpdate(g2, sg2, lr, 0.0, 1.0);

///			// TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
///			ASSERT_TRUE(p1.IsEqualTo(p2, c_epsilonFloatE5));

///			if (deviceId != CPUDEVICE)
///				continue;

///			// GPU version of SGDUpdate scales gradient by the learning rate, this check will fail.
///			// TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
///			ASSERT_TRUE(g1.IsEqualTo(g2, c_epsilonFloatE5));
///		}

///		lr = std::pow(lr, lr);
///	}
///}

///TEST_F(MatrixTests, MomentumSGDUpdate_WithAndWithout_UnitGain)
///{
///	const float low = -1.0f;
///	const float high = 1.0f;
///	float lr = 0.77f;
///	int cpuCount = 0;
///	RandomSeedFixture rsf;
///	for (auto deviceId : {CPUDEVICE, c_deviceIdZero})
///	{
///		if (deviceId<0 && ++cpuCount==2) break;

///		auto p1 = SingleMatrix::RandomUniform(12, 13, deviceId, low, high, rsf.IncrementCounter());
///		auto p2 = p1.DeepClone();
///		ASSERT_TRUE(p1.IsEqualTo(p2));

///		auto g1 = SingleMatrix::RandomUniform(12, 13, deviceId, low, high, rsf.IncrementCounter());
///		auto g2 = g1.DeepClone();
///		ASSERT_TRUE(g1.IsEqualTo(g2));

///		auto sg1 = SingleMatrix::RandomUniform(12, 13, deviceId, low, high, rsf.IncrementCounter());
///		auto sg2 = sg1.DeepClone();
///		ASSERT_TRUE(sg1.IsEqualTo(sg2));

///		for (; lr > 0.01; lr = lr / 2)
///		{
///			p1.MomentumSGDUpdate(g1, sg1, lr, 0.0, 1.0);
///			p2.MomentumSGDUpdate(g2, sg2, lr, 0.0, 1.0);
///			// TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
///			ASSERT_TRUE(p1.IsEqualTo(p2, c_epsilonFloatE5));
///		}

///		for (lr = 1.0; lr > 0.03; lr = lr / 2)
///		{
///			p1.MomentumSGDUpdate(g1, sg1, lr, 0.5, 0.5);
///			p2.MomentumSGDUpdate(g2, sg2, lr/2, 0.5, 1.0);
///			// TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
///			ASSERT_TRUE(p1.IsEqualTo(p2, c_epsilonFloatE5));
///		}

///		// TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
///		ASSERT_TRUE(g1.IsEqualTo(g2, c_epsilonFloatE5));
///		ASSERT_TRUE(sg1.IsEqualTo(sg2, c_epsilonFloatE5));

///		p1.MomentumSGDUpdate(g1, sg1, lr, 0.5, 0.5);
///		p2.MomentumSGDUpdate(g2, sg2, lr, 0.5, 1.0);
///		// TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
///		ASSERT_TRUE(!p1.IsEqualTo(p2, c_epsilonFloatE5));

///		lr = std::pow(lr, lr);
///	}
///}

///TEST_F(MatrixTests, NesterovAcceleratedMomentumSGDUpdate_WithAndWithout_UnitGain)
///{
///	const float low = -1.0f;
///	const float high = 1.0f;
///	float lr = 0.77f;
///	int cpuCount = 0;
///	RandomSeedFixture rsf;
///	for (auto deviceId : {CPUDEVICE, c_deviceIdZero})
///	{
///		if (deviceId<0 && ++cpuCount==2) break;

///		auto p1 = SingleMatrix::RandomUniform(12, 13, deviceId, low, high, rsf.IncrementCounter());
///		auto p2 = p1.DeepClone();
///		ASSERT_TRUE(p1.IsEqualTo(p2));

///		auto g1 = SingleMatrix::RandomUniform(12, 13, deviceId, low, high, rsf.IncrementCounter());
///		auto g2 = g1.DeepClone();
///		ASSERT_TRUE(g1.IsEqualTo(g2));

///		auto sg1 = SingleMatrix::RandomUniform(12, 13, deviceId, low, high, rsf.IncrementCounter());
///		auto sg2 = sg1.DeepClone();
///		ASSERT_TRUE(sg1.IsEqualTo(sg2));

///		for (; lr > 0.01; lr = lr / 2)
///		{
///			p1.NesterovAcceleratedMomentumSGDUpdate(g1, sg1, lr, 0.0, 1.0);
///			p2.NesterovAcceleratedMomentumSGDUpdate(g2, sg2, lr, 0.0, 1.0);
///			// TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
///			ASSERT_TRUE(p1.IsEqualTo(p2, c_epsilonFloatE5));
///		}

///		for (lr = 1.0; lr > 0.03; lr = lr / 2)
///		{
///			p1.NesterovAcceleratedMomentumSGDUpdate(g1, sg1, lr, 0.5, 0.5);
///			p2.NesterovAcceleratedMomentumSGDUpdate(g2, sg2, lr/2, 0.5, 1.0);
///			// TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
///			ASSERT_TRUE(p1.IsEqualTo(p2, c_epsilonFloatE5));
///		}

///		// TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
///		ASSERT_TRUE(g1.IsEqualTo(g2));
///		ASSERT_TRUE(sg1.IsEqualTo(sg2));

///		p1.NesterovAcceleratedMomentumSGDUpdate(g1, sg1, lr, 0.5, 0.5);
///		p2.NesterovAcceleratedMomentumSGDUpdate(g2, sg2, lr, 0.5, 1.0);

///		// TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
///		ASSERT_TRUE(!p1.IsEqualTo(p2, c_epsilonFloatE5));

///		lr = std::pow(lr, lr);
///	}
///}

///TEST_F(MatrixTests, FSAdagradUpdate_WithAndWithout_UnitGain)
///{
///	const float low = -1.0f;
///	const float high = 1.0f;
///	float lr = 0.77f;
///	int cpuCount = 0;
///	RandomSeedFixture rsf;
///	for (auto deviceId : {CPUDEVICE, c_deviceIdZero})
///	{
///		if (deviceId<0 && ++cpuCount==2) break;

///		auto p1 = SingleMatrix::RandomUniform(12, 13, deviceId, low, high, rsf.IncrementCounter());
///		auto p2 = p1.DeepClone();
///		ASSERT_TRUE(p1.IsEqualTo(p2));

///		auto g1 = SingleMatrix::RandomUniform(12, 13, deviceId, low, high, rsf.IncrementCounter());
///		auto g2 = g1.DeepClone();
///		ASSERT_TRUE(g1.IsEqualTo(g2));

///		auto sg1 = SingleMatrix::RandomUniform(12, 13, deviceId, low, high, rsf.IncrementCounter());
///		auto sg2 = sg1.DeepClone();
///		ASSERT_TRUE(sg1.IsEqualTo(sg2));

///		for (; lr > 0.01; lr = lr / 2)
///		{
///			double smoothedCount = 10 / lr;
///			double targetAdagradAvDenom = 1.0;
///			double varMomentum = 1.0 - lr;
///			double targetAdagradAvDenom_x_sqrtAdagradSqrFrames = targetAdagradAvDenom * sqrt(smoothedCount);

///			sg1.FSAdagradUpdate(g1, p1, targetAdagradAvDenom_x_sqrtAdagradSqrFrames, lr, 0.0, varMomentum, 1.0);
///			sg2.FSAdagradUpdate(g2, p2, targetAdagradAvDenom_x_sqrtAdagradSqrFrames, lr, 0.0, varMomentum, 1.0 /*false*/);
///			// BUGBUG: at the moment this fails even with identical arguments.
///			// ASSERT_TRUE(p1.IsEqualTo(p2, c_epsilonFloatE5));
///		}

///		sg2.SetValue(sg1);
///		ASSERT_TRUE(sg1.IsEqualTo(sg2));

///		for (lr = 1.0; lr > 0.03; lr = lr / 2)
///		{
///			double smoothedCount = 10 / lr;
///			double targetAdagradAvDenom = 1.0;
///			double varMomentum = 1.0 - lr;
///			double targetAdagradAvDenom_x_sqrtAdagradSqrFrames = targetAdagradAvDenom * sqrt(smoothedCount);

///			sg1.FSAdagradUpdate(g1, p1, targetAdagradAvDenom_x_sqrtAdagradSqrFrames, lr, 0.5, varMomentum, 0.5);
///			sg2.FSAdagradUpdate(g2, p2, targetAdagradAvDenom_x_sqrtAdagradSqrFrames, lr /*lr/2*/, 0.5, varMomentum, 0.5 /*false*/);
///			// BUGBUG: at the moment this fails even with identical arguments.
///			// ASSERT_TRUE(p1.IsEqualTo(p2, c_epsilonFloatE5));
///		}

///		lr = std::pow(lr, lr);
///	}
///}

class HalfMatrixTests : public ::testing::Test
{
public:
	static void SetUpTestCase() {}
	static void TearDownTestCase() {}
};

///TEST_F(HalfMatrixTests, Constructors)
///{
///	HalfMatrix a0(0);
///	HalfMatrix a1(0);
///	HalfMatrix a2(CPUDEVICE);
///	HalfMatrix a3(13, 12, c_deviceIdZero);

///	ASSERT_EQ(0, a0.GetNumRows());
///	ASSERT_EQ(0, a0.GetNumCols());
///	ASSERT_EQ(0, a1.GetNumRows());
///	ASSERT_EQ(0, a1.GetNumCols());
///	ASSERT_EQ(13, a3.GetNumRows());
///	ASSERT_EQ(12, a3.GetNumCols());

///	ASSERT_EQ(a0.GetDeviceId(), c_deviceIdZero);
///	ASSERT_EQ(a1.GetDeviceId(), c_deviceIdZero);
///	ASSERT_EQ(a2.GetDeviceId(), CPUDEVICE);
///	ASSERT_EQ(a3.GetDeviceId(), c_deviceIdZero);
///}

///TEST_F(HalfMatrixTests, MoveTest1)
///{
///	// no moves required
///	HalfMatrix a(c_deviceIdZero);
///	HalfMatrix b(c_deviceIdZero);
///	b.Resize(50, 100);

///	ASSERT_EQ(b.GetNumRows(), 50);
///	ASSERT_EQ(b.GetNumCols(), 100);
///	ASSERT_EQ(a.GetDeviceId(), c_deviceIdZero);
///	ASSERT_EQ(b.GetDeviceId(), c_deviceIdZero);

///	std::swap(a, b);
///	ASSERT_EQ(a.GetNumRows(), 50);
///	ASSERT_EQ(a.GetNumCols(), 100);
///	ASSERT_EQ(a.GetDeviceId(), c_deviceIdZero);
///	ASSERT_EQ(b.GetDeviceId(), c_deviceIdZero);
///}

///TEST_F(HalfMatrixTests, MoveTest2)
///{
///	// potentially a move is required
///	HalfMatrix a(c_deviceIdZero);
///	HalfMatrix b(c_deviceIdZero);
///	b.Resize(50, 100);
///	ASSERT_EQ(b.GetNumRows(), 50);
///	ASSERT_EQ(b.GetNumCols(), 100);
///	ASSERT_EQ(a.GetDeviceId(), c_deviceIdZero);
///	ASSERT_EQ(b.GetDeviceId(), c_deviceIdZero);

///	b(12, 13) = 14; // this will move whole matrix B from GPU to CPU
///	ASSERT_EQ(b.GetDeviceId(), -1);

///	std::swap(a, b); // this will not only swap A and B but will put them to their preferred device (GPU if present)
///	ASSERT_EQ(a.GetNumRows(), 50);
///	ASSERT_EQ(a.GetNumCols(), 100);
///	ASSERT_EQ(b.GetNumRows(), 0);
///	ASSERT_EQ(b.GetNumCols(), 0);
///	ASSERT_EQ(a.GetDeviceId(), -1);
///	ASSERT_EQ(b.GetDeviceId(), c_deviceIdZero);
///}

///TEST_F(HalfMatrixTests, DeepCopy)
///{
///	// This is deep copy, not move
///	HalfMatrix a(c_deviceIdZero);
///	HalfMatrix b(c_deviceIdZero);

///	b.Resize(50, 100);
///	ASSERT_EQ(a.GetNumRows(), 0);
///	ASSERT_EQ(a.GetNumCols(), 0);
///	ASSERT_EQ(b.GetNumRows(), 50);
///	ASSERT_EQ(b.GetNumCols(), 100);

///	b.SetValue(a);
///	ASSERT_EQ(a.GetNumRows(), 0);
///	ASSERT_EQ(a.GetNumCols(), 0);
///	ASSERT_EQ(b.GetNumRows(), 0);
///	ASSERT_EQ(b.GetNumCols(), 0);

///	b.Resize(50, 100);
///	ASSERT_EQ(a.GetNumRows(), 0);
///	ASSERT_EQ(a.GetNumCols(), 0);
///	ASSERT_EQ(b.GetNumRows(), 50);
///	ASSERT_EQ(b.GetNumCols(), 100);

///	b(2, 3) = 9;
///	ASSERT_EQ(b(2, 3), 9);

///	b.SetValue(a);
///	ASSERT_EQ(a.GetNumRows(), 0);
///	ASSERT_EQ(a.GetNumCols(), 0);
///	ASSERT_EQ(b.GetNumRows(), 0);
///	ASSERT_EQ(b.GetNumCols(), 0);
///}

///TEST_F(HalfMatrixTests, InitZero)
///{
///	HalfMatrix a = HalfMatrix::Zeros(12, 32, c_deviceIdZero);
///	ASSERT_EQ(a.GetNumRows(), 12);
///	ASSERT_EQ(a.GetNumCols(), 32);
///	foreach_coord (i, j, a)
///	{
///		ASSERT_EQ(a(i,j), 0.0);
///	}
///}

///TEST_F(HalfMatrixTests, InitEye)
///{
///	HalfMatrix a = HalfMatrix::Eye(56, c_deviceIdZero);
///	ASSERT_EQ(a.GetNumRows(), 56);
///	ASSERT_EQ(a.GetNumCols(), 56);

///	foreach_coord (i, j, a)
///	{
///		if (i != j)
///		{
///			ASSERT_EQ(a(i,j), 0.0);
///		}
///		else
///		{
///			ASSERT_EQ(a(i,j), 1.0);
///		}
///	}
///}

///TEST_F(HalfMatrixTests, InitOnes)
///{
///	HalfMatrix a = HalfMatrix::Ones(12, 56, c_deviceIdZero);
///	ASSERT_EQ(a.GetNumRows(), 12);
///	ASSERT_EQ(a.GetNumCols(), 56);
///	foreach_coord (i, j, a)
///	{
///		ASSERT_EQ(a(i,j), 1.0);
///	}
///}

///TEST_F(HalfMatrixTests, InitGaussianRand)
///{
///	RandomSeedFixture rsf;
///	HalfMatrix a = HalfMatrix::RandomGaussian(640, 230, c_deviceIdZero, 0.0f, 2.0f, rsf.IncrementCounter());
///	ASSERT_EQ(a.GetNumRows(), 640);
///	ASSERT_EQ(a.GetNumCols(), 230);
///	float avg = 0;
///	foreach_coord (i, j, a)
///	{
///		avg += (float)a(i,j);
///	}
///	avg /= (640 * 230);

///	float std = 0;
///	foreach_coord (i, j, a)
///	{
///		std += ((a(i,j) - avg) * (a(i,j) - avg));
///	}
///	std = sqrt(std / (640 * 230));

///	ASSERT_LE(fabs(avg), c_epsilonFloatE1);
///	ASSERT_LE(fabs(std - 2), c_epsilonFloatE1);
///}

///TEST_F(HalfMatrixTests, InitRandomUniform)
///{
///	const half low = -26.3f;
///	const half high = 30.2f;
///	RandomSeedFixture rsf;
///	HalfMatrix a = HalfMatrix::RandomUniform(435, 100, c_deviceIdZero, low, high, rsf.IncrementCounter());
///	bool has_small = false;
///	bool has_big = false;
///	foreach_coord (i, j, a)
///	{
///		ASSERT_GE(a(i,j), low);
///		ASSERT_LE(a(i,j), high);
///		if (a(i,j) < -3)
///		{
///			has_small = true;
///		}
///		if (a(i,j) > 3)
///		{
///			has_big = true;
///		}
///	}
///	ASSERT_TRUE(has_small);
///	ASSERT_TRUE(has_big);
///}

///TEST_F(HalfMatrixTests, InitRandomUniformSeed)
///{
///	const half low = -0.01f;
///	const half high = 0.01f;
///	RandomSeedFixture rsf;
///	HalfMatrix a = HalfMatrix::RandomUniform(429, 1024, c_deviceIdZero, low, high, rsf.IncrementCounter());
///	foreach_coord (i, j, a)
///	{
///		ASSERT_GE(a(i,j), low);
///		ASSERT_LE(a(i,j), high);
///	}

///	// HalfMatrix b = HalfMatrix::RandomUniform(429, 1024, (float)-0.01, (float) 0.01, rsf.IncrementCounter());
///	// ASSERT_TRUE(a.IsEqualTo(b));
///}

///TEST_F(HalfMatrixTests, SetValueMethods)
///{
///	// void SetValue(const ElemType v);
///	HalfMatrix a(32, 12, c_deviceIdZero);
///	ASSERT_EQ(32, a.GetNumRows());
///	ASSERT_EQ(12, a.GetNumCols());
///	ASSERT_EQ(12 * 32, a.GetNumElements());
///	const half v = -32.3451f;
///	a.SetValue(v);
///	foreach_coord (i, j, a)
///	{
///		ASSERT_EQ(v, a(i,j));
///	}

///	// void SetValue(const Matrix<ElemType>& deepCopyFrom);
///	HalfMatrix b(c_deviceIdZero);
///	b.SetValue(a);
///	foreach_coord (i, j, b)
///	{
///		ASSERT_EQ(v, b(i,j));
///	}

///	// void SetValue(const size_t numRows, const size_t numCols, ElemType *pArray, const bool srcIsColMajor);
///	std::array<half, 7> arrVector = { 123.0f, 0.23f, -22.0f, 63.0f, 43.42f, 324.3f, 99912.0f };

///	half *arr = arrVector.data();
///	b.SetValue(2, 3, b.GetDeviceId(), arr, matrixFlagNone);

///	HalfMatrix b1(c_deviceIdZero);
///	b1.SetValue(2, 3, b.GetDeviceId(), arr);
///	foreach_coord (i, j, b1)
///	{
///		ASSERT_EQ(arr[IDX2C(i, j, 2)], b(i,j));
///		ASSERT_EQ(arr[IDX2C(i, j, 2)], b1(i,j));
///	}

///	HalfMatrix bbbb = HalfMatrix::Zeros(6, 8, c_deviceIdZero);
///	bbbb.SetColumn(arr, 3);
///	for (int i = 0; i < 6; ++i)
///	{
///		ASSERT_EQ(arr[i], bbbb(i, 3));
///	}

///	// void SetDiagonalValue(const ElemType v);
///	HalfMatrix c(4, 4, c_deviceIdZero);
///	const half val = -0.00332f;
///	c.SetDiagonalValue(val);
///	foreach_coord (i, j, c)
///	{
///		if (i == j)
///			ASSERT_EQ(val, c(i,j));
///		else
///			ASSERT_EQ(0, c(i,j));
///	}

///	// void SetDiagonalValue(const Matrix<ElemType>& vector);
///	HalfMatrix d(4, 1, c_deviceIdZero);
///	const half val1 = 43.324f;
///	d.SetValue(val1);
///	c.SetDiagonalValue(d);
///	foreach_coord (i, j, c)
///	{
///		if (i == j)
///			ASSERT_EQ(val1, c(i,j));
///		else
///			ASSERT_EQ(0, c(i,j));
///	}

///	HalfMatrix c1(5, 5, c_deviceIdZero);
///	HalfMatrix d1(1, 5, c_deviceIdZero);
///	half val2 = 0.53f;
///	d1 = d1.Transpose();
///	d1.SetValue(val2);
///	c1.SetDiagonalValue(d1);
///	foreach_coord (i, j, c1)
///	{
///		if (i == j)
///			ASSERT_EQ(val2, c1(i,j));
///		else
///			ASSERT_EQ(0, c1(i,j));
///	}
///}

///TEST_F(HalfMatrixTests, TransposeTest)
///{
///	RandomSeedFixture rsf;
///	HalfMatrix a = HalfMatrix::RandomGaussian(64, 23, c_deviceIdZero, 0, 2, rsf.IncrementCounter());
///	ASSERT_EQ(64, a.GetNumRows());
///	ASSERT_EQ(23, a.GetNumCols());

///	HalfMatrix b = a.Transpose();

///	ASSERT_EQ(23, b.GetNumRows());
///	ASSERT_EQ(64, b.GetNumCols());

///	foreach_coord (i, j, a)
///	{
///		ASSERT_EQ(a(i,j), b(j, i));
///	}
///}

///TEST_F(HalfMatrixTests, MultiAndDiv)
///{
///	HalfMatrix m0(2, 3, c_deviceIdZero);
///	m0(0, 0) = 1;
///	m0(0, 1) = 2;
///	m0(0, 2) = 3;
///	m0(1, 0) = 4;
///	m0(1, 1) = 5;
///	m0(1, 2) = 6;

///	HalfMatrix m00(2, 3, c_deviceIdZero);
///	m00(0, 0) = 10;
///	m00(0, 1) = 20;
///	m00(0, 2) = 30;
///	m00(1, 0) = 40;
///	m00(1, 1) = 50;
///	m00(1, 2) = 60;

///	HalfMatrix m1(2, 3, c_deviceIdZero);
///	m1.Reshape(3, 2);
///	m1(0, 0) = 11;
///	m1(0, 1) = 15;
///	m1(1, 0) = 14;
///	m1(1, 1) = 13;
///	m1(2, 0) = 12;
///	m1(2, 1) = 16;

///	HalfMatrix m2(2, 2, c_deviceIdZero);
///	m2(0, 0) = 75;
///	m2(0, 1) = 89;
///	m2(1, 0) = 186;
///	m2(1, 1) = 221;

///	HalfMatrix m3 = m0 * m1;
///	ASSERT_TRUE(m3.IsEqualTo(m2));

///	m3 = m0 * 10;
///	ASSERT_TRUE(m3.IsEqualTo(m00));

///	m3 = m3 / 10;
///	ASSERT_TRUE(m3.IsEqualTo(m0));

///	m3 *= 10;
///	ASSERT_TRUE(m3.IsEqualTo(m00));

///	m3 /= 10;
///	ASSERT_TRUE(m3.IsEqualTo(m0));

///	HalfMatrix::MultiplyAndWeightedAdd(1, m0, false, m1, false, 0, m3);
///	ASSERT_TRUE(m3.IsEqualTo(m2));

///	m1.Reshape(2, 3);
///	HalfMatrix::MultiplyAndWeightedAdd(1, m0, false, m1, true, 0, m3);
///	m2(0, 0) = 74;
///	m2(0, 1) = 92;
///	m2(1, 0) = 182;
///	m2(1, 1) = 227;
///	ASSERT_TRUE(m3.IsEqualTo(m2));

///	HalfMatrix::MultiplyAndWeightedAdd(10, m0, false, m1, true, 2, m3);
///	m2(0, 0) = 888;
///	m2(0, 1) = 1104;
///	m2(1, 0) = 2184;
///	m2(1, 1) = 2724;
///	ASSERT_TRUE(m3.IsEqualTo(m2));

///	HalfMatrix::MultiplyAndWeightedAdd(1, m0, true, m1, false, 0, m3);
///	m2.Resize(3, 3);
///	m2(0, 0) = 67;
///	m2(0, 1) = 72;
///	m2(0, 2) = 77;
///	m2(1, 0) = 92;
///	m2(1, 1) = 99;
///	m2(1, 2) = 106;
///	m2(2, 0) = 117;
///	m2(2, 1) = 126;
///	m2(2, 2) = 135;
///	ASSERT_TRUE(m3.IsEqualTo(m2));

///	// Multiplications of arbitrary matrix with 1x1 matrix

///	HalfMatrix a(2, 3, c_deviceIdZero);
///	a(0, 0) = 1;
///	a(0, 1) = 2;
///	a(0, 2) = 3;
///	a(1, 0) = 4;
///	a(1, 1) = 5;
///	a(1, 2) = 6;

///	HalfMatrix b = HalfMatrix::Eye(1, c_deviceIdZero);

///	HalfMatrix c = a * b;
///	ASSERT_TRUE(c.IsEqualTo(a));
///	c = b * a;
///	ASSERT_TRUE(c.IsEqualTo(a));

///	b(0, 0) = 0.5;
///	b.InplaceAbs();
///	c = a * b;

///	HalfMatrix d(2, 3, c_deviceIdZero);
///	d(0, 0) = 0.5;
///	d(0, 1) = 1;
///	d(0, 2) = 1.5;
///	d(1, 0) = 2;
///	d(1, 1) = 2.5;
///	d(1, 2) = 3;
///	ASSERT_TRUE(c.IsEqualTo(d));
///}

///TEST_F(HalfMatrixTests, Transpose)
///{
///	HalfMatrix m0(2, 3, c_deviceIdZero);
///	m0(0, 0) = 1;
///	m0(0, 1) = 2;
///	m0(0, 2) = 3;
///	m0(1, 0) = 4;
///	m0(1, 1) = 5;
///	m0(1, 2) = 6;

///	HalfMatrix m1(3, 2, c_deviceIdZero);
///	m1(0, 0) = 1;
///	m1(0, 1) = 4;
///	m1(1, 0) = 2;
///	m1(1, 1) = 5;
///	m1(2, 0) = 3;
///	m1(2, 1) = 6;

///	HalfMatrix m2 = m0.Transpose();
///	ASSERT_TRUE(m2.IsEqualTo(m1, c_epsilonFloatE4));

///	m2.AssignTransposeOf(m1);
///	ASSERT_TRUE(m2.IsEqualTo(m0, c_epsilonFloatE4));
///}

///TEST_F(HalfMatrixTests, AddAndSub)
///{
///	HalfMatrix m0(2, 3, c_deviceIdZero);
///	m0(0, 0) = 1;
///	m0(0, 1) = 2;
///	m0(0, 2) = 3;
///	m0(1, 0) = 4;
///	m0(1, 1) = 5;
///	m0(1, 2) = 6;
///	m0.TransferFromDeviceToDevice(-1, c_deviceIdZero); // transfer to GPU for half test since CPU not working yet

///	HalfMatrix m1(2, 3, c_deviceIdZero);
///	m1(0, 0) = 11;
///	m1(0, 1) = 12;
///	m1(0, 2) = 13;
///	m1(1, 0) = 14;
///	m1(1, 1) = 15;
///	m1(1, 2) = 16;
///	m1.TransferFromDeviceToDevice(-1, c_deviceIdZero);

///	HalfMatrix m2(2, 3, c_deviceIdZero);
///	m2(0, 0) = 12;
///	m2(0, 1) = 14;
///	m2(0, 2) = 16;
///	m2(1, 0) = 18;
///	m2(1, 1) = 20;
///	m2(1, 2) = 22;
///	m2.TransferFromDeviceToDevice(-1, c_deviceIdZero);

///	HalfMatrix m3 = m2 - m0;

///	ASSERT_EQ(m3.GetDeviceId(), c_deviceIdZero);
///	ASSERT_TRUE(m3.IsEqualTo(m1, 1e-5));

///	m3 += m0;
///	ASSERT_TRUE(m3.IsEqualTo(m2));

///	m3 = m0 + 10;
///	ASSERT_TRUE(m3.IsEqualTo(m1));

///	m3 -= 10;
///	ASSERT_TRUE(m3.IsEqualTo(m0));

///	m3 = m1 + m0;
///	ASSERT_TRUE(m3.IsEqualTo(m2, 1e-5));
///	HalfMatrix m4 = HalfMatrix::Eye(3, c_deviceIdZero);

///	m3 -= m0;
///	ASSERT_TRUE(m3.IsEqualTo(m1));

///	m3 = m1 - 10;
///	ASSERT_TRUE(m3.IsEqualTo(m0));

///	HalfMatrix m33(m3.DeepClone());
///	m3 += 10;
///	ASSERT_TRUE(m3.IsEqualTo(m1));

///	HalfMatrix m55 = HalfMatrix::Eye(1, c_deviceIdZero);
///	m55(0, 0) = 10;
///	m55.InplaceAbs();
///	m33 += m55;
///	ASSERT_TRUE(m33.IsEqualTo(m1));
///	m33 -= 10;
///	m33 = m33 + 10;
///	ASSERT_TRUE(m33.IsEqualTo(m1));
///}

///TEST_F(HalfMatrixTests, ElementOps)
///{
///	HalfMatrix m0(2, 3, c_deviceIdZero);
///	m0(0, 0) = 1;
///	m0(0, 1) = 2;
///	m0(0, 2) = 3;
///	m0(1, 0) = 4;
///	m0(1, 1) = 5;
///	m0(1, 2) = 6;
///	m0.TransferFromDeviceToDevice(-1, c_deviceIdZero); // transfer to GPU for half test since CPU not working yet

///	HalfMatrix m00(2, 3, c_deviceIdZero);
///	m00(0, 0) = 1.0f;
///	m00(0, 1) = static_cast<float>(1 / 2.0);
///	m00(0, 2) = static_cast<float>(1 / 3.0);
///	m00(1, 0) = static_cast<float>(1 / 4.0);
///	m00(1, 1) = static_cast<float>(1 / 5.0);
///	m00(1, 2) = static_cast<float>(1 / 6.0);
///	m00.TransferFromDeviceToDevice(-1, c_deviceIdZero);

///	HalfMatrix m1(2, 3, c_deviceIdZero);
///	m1(0, 0) = 1;
///	m1(0, 1) = 1;
///	m1(0, 2) = 1;
///	m1(1, 0) = 1;
///	m1(1, 1) = 1;
///	m1(1, 2) = 1;
///	m1.TransferFromDeviceToDevice(-1, c_deviceIdZero);

///	HalfMatrix m3(c_deviceIdZero);
///	m3.AssignElementProductOf(m0, m00);
///	ASSERT_TRUE(m3.IsEqualTo(m1, c_epsilonFloatE4));

///	HalfMatrix m4 = HalfMatrix::Zeros(2, 3, c_deviceIdZero);
///	m4.SetValue(m4.AddElementProductOf(m0, m00));
///	ASSERT_TRUE(m4.IsEqualTo(m1, c_epsilonFloatE4));

///	m3 = m0 ^ 4;
///	HalfMatrix m2(2, 3, c_deviceIdZero);
///	m2(0, 0) = 1;
///	m2(0, 1) = 16;
///	m2(0, 2) = 81;
///	m2(1, 0) = 256;
///	m2(1, 1) = 625;
///	m2(1, 2) = 1296;
///	m2.TransferFromDeviceToDevice(-1, c_deviceIdZero);
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE3));

///	m3.SetValue(m0);
///	m3 ^= 4;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE3));

///	m3.SetValue(m0);
///	m3.ElementMultiplyWith(m00);
///	ASSERT_TRUE(m3.IsEqualTo(m1, c_epsilonFloatE3));

///	m3.SetValue(m0);
///	m3.ElementInverse();
///	ASSERT_TRUE(m3.IsEqualTo(m00, c_epsilonFloatE3));

///	m2(0, 0) = 0.7311f;
///	m2(0, 1) = 0.8808f;
///	m2(0, 2) = 0.9526f;
///	m2(1, 0) = 0.9820f;
///	m2(1, 1) = 0.9933f;
///	m2(1, 2) = 0.9975f;
///	m2.TransferFromDeviceToDevice(-1, c_deviceIdZero); // Move m2 to GPU again
///	m3.AssignElementDivisionOf(m2, m0);
///	m2.ElementMultiplyWith(m00);
///	ASSERT_TRUE(m3.IsEqualTo(m2, 1e-3f));

///	m3.SetValue(m0);
///	m3.InplaceSigmoid();
///	m2(0, 0) = 0.7311f;
///	m2(0, 1) = 0.8808f;
///	m2(0, 2) = 0.9526f;
///	m2(1, 0) = 0.9820f;
///	m2(1, 1) = 0.9933f;
///	m2(1, 2) = 0.9975f;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));

///	m3.SetValue(m0);
///	m3.InplaceTanh();
///	m2(0, 0) = 0.7616f;
///	m2(0, 1) = 0.9640f;
///	m2(0, 2) = 0.9951f;
///	m2(1, 0) = 0.9993f;
///	m2(1, 1) = 0.9999f;
///	m2(1, 2) = 1.0000f;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));

///	m3.SetValue(m0);
///	m3.InplaceLogSoftmax(true);
///	m3.InplaceExp();
///	m2(0, 0) = 0.0474f;
///	m2(0, 1) = 0.0474f;
///	m2(0, 2) = 0.0474f;
///	m2(1, 0) = 0.9526f;
///	m2(1, 1) = 0.9526f;
///	m2(1, 2) = 0.9526f;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));

///	m3.SetValue(m0);
///	m3.InplaceLogSoftmax(false);
///	m3.InplaceExp();
///	m2(0, 0) = 0.0900f;
///	m2(0, 1) = 0.2447f;
///	m2(0, 2) = 0.6652f;
///	m2(1, 0) = 0.0900f;
///	m2(1, 1) = 0.2447f;
///	m2(1, 2) = 0.6652f;
///	ASSERT_TRUE(m3.IsEqualTo(m2, 1e-3f));

///	m3.SetValue(m0);
///	m3.InplaceHardmax(true);
///	m2(0, 0) = 0.0f;
///	m2(0, 1) = 0.0f;
///	m2(0, 2) = 0.0f;
///	m2(1, 0) = 1.0f;
///	m2(1, 1) = 1.0f;
///	m2(1, 2) = 1.0f;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));

///	m3.SetValue(m0);
///	m3.InplaceSqrt();
///	m2(0, 0) = 1.0f;
///	m2(0, 1) = 1.4142f;
///	m2(0, 2) = 1.7321f;
///	m2(1, 0) = 2.0f;
///	m2(1, 1) = 2.2361f;
///	m2(1, 2) = 2.4495f;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));

///	m3.SetValue(m0);
///	m3.InplaceExp();
///	m2(0, 0) = 2.7183f;
///	m2(0, 1) = 7.3891f;
///	m2(0, 2) = 20.0855f;
///	m2(1, 0) = 54.5982f;
///	m2(1, 1) = 148.4132f;
///	m2(1, 2) = 403.4288f;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));

///	m3.SetValue(m0);
///	m3.InplaceExp();
///	m2(0, 0) = 2.7183f;
///	m2(0, 1) = 7.3891f;
///	m2(0, 2) = 20.0855f;
///	m2(1, 0) = 54.5982f;
///	m2(1, 1) = 148.4132f;
///	m2(1, 2) = 403.4288f;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));

///	m3.InplaceLog();
///	ASSERT_TRUE(m3.IsEqualTo(m0, c_epsilonFloatE4));

///	m3.SetValue(m0);
///	m3.InplaceTruncateBottom(2);
///	m2(0, 0) = 2;
///	m2(0, 1) = 2;
///	m2(0, 2) = 3;
///	m2(1, 0) = 4;
///	m2(1, 1) = 5;
///	m2(1, 2) = 6;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE3));

///	m3.SetValue(m0);
///	m3.InplaceTruncateTop(4);
///	m2(0, 0) = 1;
///	m2(0, 1) = 2;
///	m2(0, 2) = 3;
///	m2(1, 0) = 4;
///	m2(1, 1) = 4;
///	m2(1, 2) = 4;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE3));
///}

///TEST_F(HalfMatrixTests, ColumnElementMultiply)
///{
///	// CPU path doesn't work
///	//CPUMatrix<half> mcpu = CPUMatrix<half>::RandomUniform(429, 1024, -3.4f, 1, rsf.IncrementCounter());
///	//CPUMatrix<half> acpu = CPUMatrix<half>::Ones(429, 1);
///	//CPUMatrix<half> mcpuCopy(mcpu);

///	//mcpu.ColumnElementMultiplyWith(acpu);
///	//ASSERT_TRUE(mcpuCopy.IsEqualTo(mcpu, c_epsilonFloatE4));

///	RandomSeedFixture rsf;
///	Matrix<half> m = Matrix<half>::RandomUniform(429, 1024, c_deviceIdZero, -3.4f, 1, rsf.IncrementCounter());
///	Matrix<half> a = Matrix<half>::Ones(429, 1, c_deviceIdZero);
///	Matrix<half> mCopy(m.DeepClone());

///	m.ColumnElementMultiplyWith(a);
///	ASSERT_TRUE(mCopy.IsEqualTo(m, c_epsilonFloatE4));

///	//CPUMatrix<half> mc1 = CPUMatrix<half>::RandomUniform(429, 1024, -3.4f, 1, rsf.IncrementCounter());
///	//CPUMatrix<half> mc2 = CPUMatrix<half>::RandomUniform(429, 1, 0, 3, rsf.IncrementCounter());
///	//mc1.ColumnElementMultiplyWith(mc2);

///	//Matrix<half> m1(mc1.GetNumRows(), mc1.GetNumCols(), mc1.GetBuffer(), matrixFlagNone);
///	//Matrix<half> m2(mc2.GetNumRows(), mc2.GetNumCols(), mc2.GetBuffer(), matrixFlagNone);
///	//m1.ColumnElementMultiplyWith(m2);
///	//foreach_coord (i, j, m2)
///	//{
///	//	ASSERT_LT(fabs(m2(i,j) - mc2(i,j)), c_epsilonFloatE5);
///	//}
///}

///TEST_F(HalfMatrixTests, AssignXOf)
///{
///	// AssignDifferenceOf
///	RandomSeedFixture rsf;
///	Matrix<half> a = Matrix<half>::RandomUniform(429, 1024, c_deviceIdZero, 5, 32, rsf.IncrementCounter());
///	Matrix<half> b = Matrix<half>::RandomUniform(429, 1024, c_deviceIdZero, 5, 32, rsf.IncrementCounter());
///	Matrix<half> c(c_deviceIdZero);

///	c.AssignDifferenceOf(a, b);
///	foreach_coord (i, j, c)
///	{
///		ASSERT_EQ(c(i,j), a(i,j) - b(i,j));
///	}
///	a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

///	half x = 234.2f;
///	c.AssignDifferenceOf(a, x);
///	foreach_coord (i, j, c)
///	{
///		ASSERT_EQ(c(i,j), a(i,j) - x);
///	}
///	a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

///	c.AssignDifferenceOf(x, a);
///	foreach_coord (i, j, c)
///	{
///		ASSERT_EQ(c(i,j), x - a(i,j));
///	}
///	a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

///	c.AssignDifferenceOf(1, a);
///	foreach_coord (i, j, c)
///	{
///		ASSERT_EQ(c(i,j), 1 - a(i,j));
///	}
///	a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

///	// AssignSumOf
///	c.AssignSumOf(a, b);
///	foreach_coord (i, j, c)
///	{
///		ASSERT_EQ(c(i,j), a(i,j) + b(i,j));
///	}
///	a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

///	// Check for self-assignment (c = c + b)
///	auto tolerance = 5e-5;
///	c.AssignSumOf(c, b);
///	foreach_coord (i, j, c)
///	{
///		half ab = a(i,j) + b(i,j);
///		ASSERT_NEAR(c(i,j), ab + b(i,j), tolerance);
///	}
///	a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

///	// Check for self-assignment (c = b + c)
///	c.AssignSumOf(b, c);
///	foreach_coord (i, j, c)
///	{
///		half ab = a(i,j) + b(i,j);
///		ASSERT_NEAR(c(i,j), ab + b(i,j) + b(i,j), tolerance);
///	}
///	a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

///	// Check for self-assignment (c = c + a .* c)
///	c.AssignSumOf(a, b);
///	c.AddElementProductOf(a, c);
///	foreach_coord(i, j, c)
///	{
///		half ab = a(i,j) + b(i,j);
///		ASSERT_NEAR(c(i,j), ab + a(i,j)*ab, tolerance);
///	}
///	a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

///	// Check for self-assignment (c = c + c .* a)
///	c.AssignSumOf(a, b);
///	c.AddElementProductOf(c, a);
///	foreach_coord(i, j, c)
///	{
///		half ab = a(i,j) + b(i,j);
///		ASSERT_NEAR(c(i,j), ab + a(i,j)*ab, tolerance);
///	}
///	a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

///	// Check for self-assignment (c = c + c .* c)
///	c.AssignSumOf(a, b);
///	c.AddElementProductOf(c, c);
///	foreach_coord(i, j, c)
///	{
///		half ab = a(i,j) + b(i,j);
///		ASSERT_NEAR(c(i,j), ab + ab*ab, tolerance);
///	}
///	a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

///	// AssignElementProductOf
///	c.AssignElementProductOf(a, b);
///	foreach_coord (i, j, c)
///	{
///		ASSERT_EQ(c(i,j), a(i,j) * b(i,j));
///	}
///	a.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	b.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);
///	c.TransferToDeviceIfNotThere(c_deviceIdZero, true, false, true);

///	// AddElementProductOf
///	Matrix<half> c_copy(c.DeepClone());
///	c.AddElementProductOf(a, b);
///	foreach_coord (i, j, c)
///	{
///		ASSERT_EQ(c(i,j), c_copy(i,j) + a(i,j) * b(i,j));
///	}

///	// AssignSigmoidOf
///	//CPUMatrix<half> ac = CPUMatrix<half>::RandomUniform(429, 1024, 5, 32, rsf.IncrementCounter());
///	//CPUMatrix<half> bc = CPUMatrix<half>::RandomUniform(429, 1024, -5, 12, rsf.IncrementCounter());
///	//Matrix<half> d(ac.GetNumRows(), ac.GetNumCols(), ac.GetBuffer(), matrixFlagNone);
///	//Matrix<half> e(bc.GetNumRows(), bc.GetNumCols(), bc.GetBuffer(), matrixFlagNone);
///	//ac.AssignSigmoidOf(bc);
///	//d.AssignSigmoidOf(e);
///	//foreach_coord (i, j, ac)
///	//{
///	//	ASSERT_LT(fabs(ac(i,j) - d(i,j)), c_epsilonFloatE5);
///	//}

///	// AssignSignOf
///	Matrix<half> m1 = Matrix<half>::RandomUniform(42, 12, c_deviceIdZero, -5, 12, rsf.IncrementCounter());
///	Matrix<half> m2(4, 5, c_deviceIdZero);
///	m2.AssignSignOf(m1);
///	foreach_coord (i, j, m1)
///	{
///		half v = m1(i,j);
///		half expected = SIGNUMZ(v);
///		half actual = m2(i,j);
///		ASSERT_EQ(expected, actual);
///	}

///	Matrix<half> m3 = Matrix<half>::RandomUniform(42, 12, c_deviceIdZero, -5, 2, rsf.IncrementCounter());
///	Matrix<half> m4(m3.DeepClone());
///	m3.AddSignOf(m1);
///	foreach_coord (i, j, m3)
///	{
///		half v = m1(i,j);
///		ASSERT_EQ(half(m4(i,j) + SIGNUMZ(v)), m3(i,j));
///	}

///	// AssignTruncateBottom and Top
///	Matrix<half> m5(2, 2, c_deviceIdZero);
///	m5(0, 0) = 1;
///	m5(0, 1) = 2;
///	m5(1, 0) = 3;
///	m5(1, 1) = 4;
///	m5.TransferFromDeviceToDevice(-1, c_deviceIdZero);

///	Matrix<half> m6(c_deviceIdZero);
///	m6.AssignTruncateBottomOf(m5, 3);
///	ASSERT_EQ(3, m6(0, 0));
///	ASSERT_EQ(3, m6(0, 1));
///	ASSERT_EQ(3, m6(1, 0));
///	ASSERT_EQ(4, m6(1, 1));

///	Matrix<half> m7(c_deviceIdZero);
///	m7.AssignTruncateTopOf(m5, 3);
///	ASSERT_EQ(1, m7(0, 0));
///	ASSERT_EQ(2, m7(0, 1));
///	ASSERT_EQ(3, m7(1, 0));
///	ASSERT_EQ(3, m7(1, 1));
///}

///TEST_F(HalfMatrixTests, SumOfElements)
///{
///	Matrix<half> m = Matrix<half>::Ones(429, 1, c_deviceIdZero);
///	half sum = m.SumOfElements();
///	ASSERT_EQ(429, sum);

///	//CPUMatrix<half> mcpu = CPUMatrix<half>::Ones(429, 1024);
///	//half sumCPU = mcpu.SumOfElements();
///	//ASSERT_EQ(429 * 1024, sumCPU);

///	Matrix<half> m1 = Matrix<half>::Ones(1, 332, c_deviceIdZero); m1 *= -1;
///	half sum1 = m1.SumOfElements();
///	ASSERT_EQ(-332, sum1);

///	Matrix<half> m2 = Matrix<half>::Ones(3, 2, c_deviceIdZero); m2 *= -1;
///	half sum2 = m2.SumOfElements();
///	ASSERT_EQ(-6, sum2);
///}

///TEST_F(HalfMatrixTests, ColumnSlice)
///{
///	std::array<half, 6> arr = { 1, 2, 3, 4, 5, 6};
///	auto *fArray = arr.data();

///	Matrix<half> m0(2, 3, fArray, c_deviceIdZero, matrixFlagNone);
///	Matrix<half> m1(2, 2, fArray, c_deviceIdZero, matrixFlagNone);
///	Matrix<half> m2 = m0.ColumnSlice(0, 2);
///	ASSERT_TRUE(m2.IsEqualTo(m1, c_epsilonFloatE4));

///	Matrix<half> m3(2, 2, fArray + 2, c_deviceIdZero, matrixFlagNone);
///	m2 = m0.ColumnSlice(1, 2);
///	ASSERT_TRUE(m2.IsEqualTo(m3, c_epsilonFloatE4));

///	RandomSeedFixture rsf;
///	size_t k = 100, n = 20, m = 50;

///	Matrix<half> ag(k, n, c_deviceIdZero);
///	ag.SetUniformRandomValue(-1, 1, rsf.IncrementCounter());

///	Matrix<half> bg(n, m, c_deviceIdZero);
///	bg.SetUniformRandomValue(-1, 1, rsf.IncrementCounter());

///	Matrix<half> cg(k, m, c_deviceIdZero);
///	cg.SetUniformRandomValue(-1, 1, rsf.IncrementCounter());

///	Matrix<half> dg(k, m, c_deviceIdZero);
///	dg.AssignValuesOf(cg);

///	Matrix<half>::MultiplyAndAdd(ag, false, bg, false, dg);
///	for (int i = 0; i < m; i++)
///	{
///		Matrix<half> colBg = bg.ColumnSlice(i, 1);
///		Matrix<half> colCg = cg.ColumnSlice(i, 1);
///		Matrix<half>::MultiplyAndAdd(ag, false, colBg, false, colCg);
///		Matrix<half> colDg = dg.ColumnSlice(i, 1);
///		ASSERT_TRUE(colCg.IsEqualTo(colDg, c_epsilonFloat5E4));
///	}
///}

///TEST_F(HalfMatrixTests, KhatriRaoProduct)
///{
///	std::array<half, 24> arr =
///		{ 0, 0, 0, 0, 0, 0,
///		0, 0, 0, 0, 0, 0,
///		0, 0, 0, 0, 0, 0,
///		0, 0, 0, 0, 0, 0};

///	auto *fArray = arr.data();
///	fArray[0] = 0.8147f;
///	fArray[3] = 0.9134f;
///	fArray[6] = 0.2785f;
///	fArray[9] = 0.9649f;
///	fArray[1] = 0.9058f;
///	fArray[4] = 0.6324f;
///	fArray[7] = 0.5469f;
///	fArray[10] = 0.1576f;
///	fArray[2] = 0.1270f;
///	fArray[5] = 0.0975f;
///	fArray[8] = 0.9575f;
///	fArray[11] = 0.9706f;
///	Matrix<half> a(3, 4, fArray, c_deviceIdZero);

///	fArray[0] = 0.9572f;
///	fArray[2] = 0.8003f;
///	fArray[4] = 0.4218f;
///	fArray[6] = 0.7922f;
///	fArray[1] = 0.4854f;
///	fArray[3] = 0.1419f;
///	fArray[5] = 0.9157f;
///	fArray[7] = 0.9595f;
///	Matrix<half> b(2, 4, fArray, c_deviceIdZero);

///	fArray[0] = 0.7798f;
///	fArray[6] = 0.7310f;
///	fArray[12] = 0.1175f;
///	fArray[18] = 0.7644f;
///	fArray[1] = 0.8670f;
///	fArray[7] = 0.5061f;
///	fArray[13] = 0.2307f;
///	fArray[19] = 0.1249f;
///	fArray[2] = 0.1215f;
///	fArray[8] = 0.0781f;
///	fArray[14] = 0.4038f;
///	fArray[20] = 0.7689f;
///	fArray[3] = 0.3954f;
///	fArray[9] = 0.1296f;
///	fArray[15] = 0.2550f;
///	fArray[21] = 0.9258f;
///	fArray[4] = 0.4396f;
///	fArray[10] = 0.0897f;
///	fArray[16] = 0.5008f;
///	fArray[22] = 0.1512f;
///	fArray[5] = 0.0616f;
///	fArray[11] = 0.0138f;
///	fArray[17] = 0.8768f;
///	fArray[23] = 0.9313f;
///	Matrix<half> d(6, 4, fArray, c_deviceIdZero);

///	Matrix<half> c(c_deviceIdZero);
///	c.AssignKhatriRaoProductOf(a, b);
///	ASSERT_TRUE(c.IsEqualTo(d, 0.003f));
///}

///TEST_F(HalfMatrixTests, AddColumnReshapeProductOf)
///{
///	std::array<half, 12> arr = {
///		0, 0, 0, 0, 0, 0,
///		0, 0, 0, 0, 0, 0
///	};
///	auto *fArray = arr.data();
///	fArray[0] = 0.6557f;
///	fArray[6] = 0.7431f;
///	fArray[1] = 0.0357f;
///	fArray[7] = 0.3922f;
///	fArray[2] = 0.8491f;
///	fArray[8] = 0.6555f;
///	fArray[3] = 0.9340f;
///	fArray[9] = 0.1712f;
///	fArray[4] = 0.6787f;
///	fArray[10] = 0.7060f;
///	fArray[5] = 0.7577f;
///	fArray[11] = 0.0318f;
///	Matrix<half> a(6, 2, fArray, c_deviceIdZero);

///	fArray[0] = 0.2769f;
///	fArray[3] = 0.8235f;
///	fArray[1] = 0.0462f;
///	fArray[4] = 0.6948f;
///	fArray[2] = 0.0971f;
///	fArray[5] = 0.3171f;
///	Matrix<half> b(3, 2, fArray, c_deviceIdZero);

///	fArray[0] = 0.2867f;
///	fArray[2] = 1.2913f;
///	fArray[1] = 0.1266f;
///	fArray[3] = 0.4520f;
///	Matrix<half> d0(2, 2, fArray, c_deviceIdZero);

///	fArray[0] = 0.2657f;
///	fArray[2] = 1.0923f;
///	fArray[1] = 0.3636f;
///	fArray[3] = 0.6416f;
///	Matrix<half> d1(2, 2, fArray, c_deviceIdZero);

///	Matrix<half> c(2, 2, c_deviceIdZero);
///	c.SetValue(0.0f);
///	c.AddColumnReshapeProductOf(a, b, false);
///	ASSERT_TRUE(c.IsEqualTo(d0, c_epsilonFloat3E4));

///	c.SetValue(0.0f);
///	c.AddColumnReshapeProductOf(a, b, true);
///	ASSERT_TRUE(c.IsEqualTo(d1, c_epsilonFloat3E4));
///}

///TEST_F(HalfMatrixTests, Copy)
///{
///	// Matrices are stored as column-major so below is 4x3 matrix.
///	const size_t crow = 4;
///	const size_t ccol = 3;
///	std::vector<half> src = {
///		1.0f, 3.0f, 4.0f, 7.0f,
///		6.0f, 2.0f, 5.0f, 8.0f,
///		4.0f, 9.0f, 3.0f, 1.0f };
///	HalfMatrix srcM(crow, ccol, src.data(), c_deviceIdZero, matrixFlagNone);

///	// Test full copy
///	CPUHalfMatrix actM(crow, ccol);
///	srcM.CopySection(actM.GetNumRows(), actM.GetNumCols(), actM.Data(), actM.GetNumRows());
///	ASSERT_TRUE(actM.IsEqualTo(CPUMatrix<half>(actM.GetNumRows(), actM.GetNumCols(), src.data(), matrixFlagNone)));

///	// Test tile copy
///	actM.Resize(crow-1, ccol-1);
///	actM.SetValue(std::numeric_limits<float>::quiet_NaN());
///	srcM.CopySection(actM.GetNumRows(), actM.GetNumCols(), actM.Data(), actM.GetNumRows());

///	vector<half> expected = { 1.0f, 3.0f, 4.0f, 6.0f, 2.0f, 5.0f };
///	ASSERT_TRUE(actM.IsEqualTo(CPUMatrix<half>(actM.GetNumRows(), actM.GetNumCols(), expected.data(), matrixFlagNone)));
///}

///TEST_F(HalfMatrixTests, HasElement)
///{
///	int cpuCount = 0;
///	for (auto deviceId : {CPUDEVICE, c_deviceIdZero})
///	{
///		if (deviceId<0 && ++cpuCount==2) break;

///		const size_t size = 3;
///		half src[size] = { 0.0f, 1.0f, 2.0f };
///		HalfMatrix m1(1, size, src, deviceId, matrixFlagNone);
///		ASSERT_TRUE(HalfMatrix::HasElement(m1, 1.0f));
///		ASSERT_TRUE(!HalfMatrix::HasElement(m1, -1.0f));

///		auto qnan = std::numeric_limits<float>::quiet_NaN();
///		ASSERT_TRUE(!HalfMatrix::HasElement(m1, qnan));
///		auto posInf = std::numeric_limits<float>::infinity();
///		ASSERT_TRUE(!HalfMatrix::HasElement(m1, posInf));

///		m1(0, 1) = qnan;
///		ASSERT_TRUE(HalfMatrix::HasElement(m1, qnan));

///		m1(0, 1) = posInf;
///		ASSERT_TRUE(HalfMatrix::HasElement(m1, posInf));
///	}
///}

///TEST_F(HalfMatrixTests, VectorMax)
///{
///	// Matrices are stored as column-major so below is 3x2 matrix.
///	half src[] = {
///		1.0f, 3.0f, 4.0f,
///		6.0f, 2.0f, 5.0f };

///	half expectedIdx[] = {
///		2.0f, 1.0f,
///		0.0f, 2.0f };

///	half expectedVal[] = {
///		4.0f, 3.0f,
///		6.0f, 5.0f };

///	int cpuCount = 0;
///	for (auto deviceId : {CPUDEVICE, c_deviceIdZero})
///	{
///		if (deviceId<0 && ++cpuCount==2) break;

///		Matrix<half> expIdx(2, 2, expectedIdx, deviceId, matrixFlagNone);
///		Matrix<half> expVal(2, 2, expectedVal, deviceId, matrixFlagNone);

///		Matrix<half> actual(3, 2, src, deviceId, matrixFlagNone);
///		Matrix<half> actualIdx(deviceId);
///		Matrix<half> actualVal(deviceId);

///		auto topK = 2;
///		actual.VectorMax(actualIdx, actualVal, true, topK);
///		ASSERT_TRUE(actualIdx.IsEqualTo(expIdx));
///		ASSERT_TRUE(actualVal.IsEqualTo(expVal));
///	}
///}

///TEST_F(HalfMatrixTests, AssignNumOfDiff)
///{
///	half labels[] = { 1.0f, 2.0f, 3.0f };

///	// Matrices are stored as column-major so below is 2x3 matrix.
///	half topKResults[] = {
///		1.0f, 3.0f,
///		4.0f, 6.0f,
///		2.0f, 3.0f };

///	// CPU not supported yet
///	int cpuCount = 0;
///	for (auto deviceId : {CPUDEVICE, c_deviceIdZero})
///	{
///		if (deviceId<0 && ++cpuCount==2) break;

///		Matrix<half> lbl(1, 3, labels, deviceId, matrixFlagNone);
///		Matrix<half> topKRes(2, 3, topKResults, deviceId, matrixFlagNone);

///		Matrix<half> actual(deviceId);
///		actual.AssignNumOfDiff(lbl, topKRes, true);

///		half expectedDiff = 1.0;
///		ASSERT_EQ(expectedDiff, actual.Get00Element());
///	}
///}

///TEST_F(HalfMatrixTests, Scale)
///{
///	const float low = -1.0f;
///	const float high = 1.0f;
///	float alpha = 0.7713f;
///	RandomSeedFixture rsf;
///	int cpuCount = 0;
///	for (auto deviceId : {CPUDEVICE, c_deviceIdZero})
///	{
///		if (deviceId<0 && ++cpuCount==2) break;

///		auto a1 = HalfMatrix::RandomUniform(7, 11, deviceId, low, high, rsf.IncrementCounter());
///		auto a2 = a1.DeepClone();
///		ASSERT_TRUE(a1.IsEqualTo(a2));

///		auto b1 = HalfMatrix::RandomUniform(7, 11, deviceId, low, high, rsf.IncrementCounter());
///		auto b2 = b1.DeepClone();
///		ASSERT_TRUE(b1.IsEqualTo(b2));

///		Matrix<half>::ScaleAndAdd(alpha, b1, a1);

///		Matrix<half>::Scale(alpha, b2);
///		a2 += b2;

///		// BUGBUG: this test currently fails on GPU.
///		// This not works on GPU
///		//if (deviceId != CPUDEVICE)
///		//    continue;

///		// TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
///		// ASSERT_TRUE(a1.IsEqualTo(a2));
///		ASSERT_TRUE(a1.IsEqualTo(a2, 1e-3f));
///	}
///}

///TEST_F(HalfMatrixTests, SGDUpdate)
///{
///	const float low = -1.0f;
///	const float high = 1.0f;
///	float lr = 0.77f;
///	RandomSeedFixture rsf;
///	int cpuCount = 0;
///	for (auto deviceId : {CPUDEVICE, c_deviceIdZero})
///	{
///		if (deviceId<0 && ++cpuCount==2) break;

///		auto p1 = HalfMatrix::RandomUniform(12, 13, deviceId, low, high, rsf.IncrementCounter());
///		auto p2 = p1.DeepClone();
///		ASSERT_TRUE(p1.IsEqualTo(p2));

///		auto g1 = HalfMatrix::RandomUniform(12, 13, deviceId, low, high, rsf.IncrementCounter());
///		auto g2 = g1.DeepClone();
///		ASSERT_TRUE(g1.IsEqualTo(g2));

///		auto sg1 = HalfMatrix::RandomUniform(12, 13, deviceId, low, high, rsf.IncrementCounter());
///		auto sg2 = sg1.DeepClone();
///		ASSERT_TRUE(sg1.IsEqualTo(sg2));

///		for (; lr > 0.01; lr = lr / 2)
///		{
///			if (deviceId != CPUDEVICE)
///			{
///				// g1 is modified inside the GPU version of SGDUpdate, restore the original value here.
///				g1.SetValue(g2);
///			}

///			p1.SGDUpdate(g1, lr);
///			p2.MomentumSGDUpdate(g2, sg2, lr, 0.0, 1.0);

///			// TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
///			ASSERT_TRUE(p1.IsEqualTo(p2, c_epsilonFloatE5));

///			if (deviceId != CPUDEVICE)
///				continue;

///			// GPU version of SGDUpdate scales gradient by the learning rate, this check will fail.
///			// TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
///			ASSERT_TRUE(g1.IsEqualTo(g2, c_epsilonFloatE5));
///		}

///		lr = std::pow(lr, lr);
///	}
///}

///TEST_F(HalfMatrixTests, MomentumSGDUpdate_WithAndWithout_UnitGain)
///{
///	const float low = -1.0f;
///	const float high = 1.0f;
///	float lr = 0.77f;
///	RandomSeedFixture rsf;
///	int cpuCount = 0;
///	for (auto deviceId : {CPUDEVICE, c_deviceIdZero})
///	{
///		if (deviceId<0 && ++cpuCount==2) break;

///		auto p1 = HalfMatrix::RandomUniform(12, 13, deviceId, low, high, rsf.IncrementCounter());
///		auto p2 = p1.DeepClone();
///		ASSERT_TRUE(p1.IsEqualTo(p2));

///		auto g1 = HalfMatrix::RandomUniform(12, 13, deviceId, low, high, rsf.IncrementCounter());
///		auto g2 = g1.DeepClone();
///		ASSERT_TRUE(g1.IsEqualTo(g2));

///		auto sg1 = HalfMatrix::RandomUniform(12, 13, deviceId, low, high, rsf.IncrementCounter());
///		auto sg2 = sg1.DeepClone();
///		ASSERT_TRUE(sg1.IsEqualTo(sg2));

///		for (; lr > 0.01; lr = lr / 2)
///		{
///			p1.MomentumSGDUpdate(g1, sg1, lr, 0.0, 1.0);
///			p2.MomentumSGDUpdate(g2, sg2, lr, 0.0, 1.0);
///			// TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
///			ASSERT_TRUE(p1.IsEqualTo(p2, c_epsilonFloatE5));
///		}

///		for (lr = 1.0; lr > 0.03; lr = lr / 2)
///		{
///			p1.MomentumSGDUpdate(g1, sg1, lr, 0.5, 0.5);
///			p2.MomentumSGDUpdate(g2, sg2, lr/2, 0.5, 1.0);
///			// TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
///			ASSERT_TRUE(p1.IsEqualTo(p2, c_epsilonFloatE5));
///		}

///		// TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
///		ASSERT_TRUE(g1.IsEqualTo(g2, c_epsilonFloatE5));
///		ASSERT_TRUE(sg1.IsEqualTo(sg2, c_epsilonFloatE5));

///		p1.MomentumSGDUpdate(g1, sg1, lr, 0.5, 0.5);
///		p2.MomentumSGDUpdate(g2, sg2, lr, 0.5, 1.0);
///		// TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
///		ASSERT_TRUE(!p1.IsEqualTo(p2, c_epsilonFloatE5));

///		lr = std::pow(lr, lr);
///	}
///}

///TEST_F(HalfMatrixTests, NesterovAcceleratedMomentumSGDUpdate_WithAndWithout_UnitGain)
///{
///	const float low = -1.0f;
///	const float high = 1.0f;
///	float lr = 0.77f;
///	RandomSeedFixture rsf;
///	int cpuCount = 0;
///	for (auto deviceId : {CPUDEVICE, c_deviceIdZero})
///	{
///		if (deviceId<0 && ++cpuCount==2) break;

///		auto p1 = HalfMatrix::RandomUniform(12, 13, deviceId, low, high, rsf.IncrementCounter());
///		auto p2 = p1.DeepClone();
///		ASSERT_TRUE(p1.IsEqualTo(p2));

///		auto g1 = HalfMatrix::RandomUniform(12, 13, deviceId, low, high, rsf.IncrementCounter());
///		auto g2 = g1.DeepClone();
///		ASSERT_TRUE(g1.IsEqualTo(g2));

///		auto sg1 = HalfMatrix::RandomUniform(12, 13, deviceId, low, high, rsf.IncrementCounter());
///		auto sg2 = sg1.DeepClone();
///		ASSERT_TRUE(sg1.IsEqualTo(sg2));

///		for (; lr > 0.01; lr = lr / 2)
///		{
///			p1.NesterovAcceleratedMomentumSGDUpdate(g1, sg1, lr, 0.0, 1.0);
///			p2.NesterovAcceleratedMomentumSGDUpdate(g2, sg2, lr, 0.0, 1.0);
///			// TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
///			ASSERT_TRUE(p1.IsEqualTo(p2, c_epsilonFloatE5));
///		}

///		for (lr = 1.0; lr > 0.03; lr = lr / 2)
///		{
///			p1.NesterovAcceleratedMomentumSGDUpdate(g1, sg1, lr, 0.5, 0.5);
///			p2.NesterovAcceleratedMomentumSGDUpdate(g2, sg2, lr/2, 0.5, 1.0);
///			// TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
///			ASSERT_TRUE(p1.IsEqualTo(p2, c_epsilonFloatE5));
///		}

///		// TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
///		ASSERT_TRUE(g1.IsEqualTo(g2));
///		ASSERT_TRUE(sg1.IsEqualTo(sg2));

///		p1.NesterovAcceleratedMomentumSGDUpdate(g1, sg1, lr, 0.5, 0.5);
///		p2.NesterovAcceleratedMomentumSGDUpdate(g2, sg2, lr, 0.5, 1.0);

///		// TODO: enable DeterministicCPUAlgorithmsFixture and use strict equality.
///		ASSERT_TRUE(!p1.IsEqualTo(p2, c_epsilonFloatE5));

///		lr = std::pow(lr, lr);
///	}
///}

///TEST_F(HalfMatrixTests, FSAdagradUpdate_WithAndWithout_UnitGain)
///{
///	const float low = -1.0f;
///	const float high = 1.0f;
///	float lr = 0.77f;
///	RandomSeedFixture rsf;
///	int cpuCount = 0;
///	for (auto deviceId : {CPUDEVICE, c_deviceIdZero})
///	{
///		if (deviceId<0 && ++cpuCount==2) break;

///		auto p1 = HalfMatrix::RandomUniform(12, 13, deviceId, low, high, rsf.IncrementCounter());
///		auto p2 = p1.DeepClone();
///		ASSERT_TRUE(p1.IsEqualTo(p2));

///		auto g1 = HalfMatrix::RandomUniform(12, 13, deviceId, low, high, rsf.IncrementCounter());
///		auto g2 = g1.DeepClone();
///		ASSERT_TRUE(g1.IsEqualTo(g2));

///		auto sg1 = HalfMatrix::RandomUniform(12, 13, deviceId, low, high, rsf.IncrementCounter());
///		auto sg2 = sg1.DeepClone();
///		ASSERT_TRUE(sg1.IsEqualTo(sg2));

///		for (; lr > 0.01; lr = lr / 2)
///		{
///			double smoothedCount = 10 / lr;
///			double targetAdagradAvDenom = 1.0;
///			double varMomentum = 1.0 - lr;
///			double targetAdagradAvDenom_x_sqrtAdagradSqrFrames = targetAdagradAvDenom * sqrt(smoothedCount);

///			sg1.FSAdagradUpdate(g1, p1, targetAdagradAvDenom_x_sqrtAdagradSqrFrames, lr, 0.0, varMomentum, 1.0);
///			sg2.FSAdagradUpdate(g2, p2, targetAdagradAvDenom_x_sqrtAdagradSqrFrames, lr, 0.0, varMomentum, 1.0 /*false*/);
///			// BUGBUG: at the moment this fails even with identical arguments.
///			// ASSERT_TRUE(p1.IsEqualTo(p2, c_epsilonFloatE5));
///		}

///		sg2.SetValue(sg1);
///		ASSERT_TRUE(sg1.IsEqualTo(sg2));

///		for (lr = 1.0; lr > 0.03; lr = lr / 2)
///		{
///			double smoothedCount = 10 / lr;
///			double targetAdagradAvDenom = 1.0;
///			double varMomentum = 1.0 - lr;
///			double targetAdagradAvDenom_x_sqrtAdagradSqrFrames = targetAdagradAvDenom * sqrt(smoothedCount);

///			sg1.FSAdagradUpdate(g1, p1, targetAdagradAvDenom_x_sqrtAdagradSqrFrames, lr, 0.5, varMomentum, 0.5);
///			sg2.FSAdagradUpdate(g2, p2, targetAdagradAvDenom_x_sqrtAdagradSqrFrames, lr /*lr/2*/, 0.5, varMomentum, 0.5 /*false*/);
///			// BUGBUG: at the moment this fails even with identical arguments.
///			// ASSERT_TRUE(p1.IsEqualTo(p2, c_epsilonFloatE5));
///		}

///		lr = std::pow(lr, lr);
///	}
///}

} } } }
