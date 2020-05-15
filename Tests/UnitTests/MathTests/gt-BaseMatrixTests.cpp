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

class BaseMatrixTests : public ::testing::Test
{
public:
	static void SetUpTestCase() {}
	static void TearDownTestCase() {}
};

TEST_F(BaseMatrixTests, ConstructorNoFlags)
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

TEST_F(BaseMatrixTests, ConstructorFlagNormal)
{
	BaseMatrix<float> m; 
	std::array<float, 6> array = {1, 2, 3, 4, 5, 6};
	m.Assign(2, 3, array.data(), matrixFlagNone);

	ASSERT_EQ(m.GetItem(0,0), 1);
	ASSERT_EQ(m.GetItem(0,1), 3);
	ASSERT_EQ(m.GetItem(0,2), 5);
	ASSERT_EQ(m.GetItem(1,0), 2);
	ASSERT_EQ(m.GetItem(1,1), 4);
	ASSERT_EQ(m.GetItem(1,2), 6);
}

TEST_F(BaseMatrixTests, ConstructorFormatRowMajor)
{
	BaseMatrix<float> m;
	std::array<float, 6> array = {7, 8, 9, 10, 11, 12};
	m.Assign(2, 3, array.data(), matrixFormatRowMajor);

	ASSERT_EQ(m.GetItem(0,0), 7);
	ASSERT_EQ(m.GetItem(0,1), 8);
	ASSERT_EQ(m.GetItem(0,2), 9);
	ASSERT_EQ(m.GetItem(1,0), 10);
	ASSERT_EQ(m.GetItem(1,1), 11);
	ASSERT_EQ(m.GetItem(1,2), 12);
}

///TEST_F(BaseMatrixTests, AddAndSub)
///{
///	BaseMatrix<float> m0(2, 3);
///	m0(0, 0) = 1;
///	m0(0, 1) = 2;
///	m0(0, 2) = 3;
///	m0(1, 0) = 4;
///	m0(1, 1) = 5;
///	m0(1, 2) = 6;
///
///	BaseMatrix<float> m1(2, 3);
///	m1(0, 0) = 11;
///	m1(0, 1) = 12;
///	m1(0, 2) = 13;
///	m1(1, 0) = 14;
///	m1(1, 1) = 15;
///	m1(1, 2) = 16;
///
///	BaseMatrix<float> m2(2, 3);
///	m2(0, 0) = 12;
///	m2(0, 1) = 14;
///	m2(0, 2) = 16;
///	m2(1, 0) = 18;
///	m2(1, 1) = 20;
///	m2(1, 2) = 22;
///
///	BaseMatrix<float> mC(2, 1);
///	mC(0, 0) = 10;
///	mC(1, 0) = 10;
///
///	BaseMatrix<float> mR(1, 3);
///	mR(0, 0) = 10;
///	mR(0, 1) = 10;
///	mR(0, 2) = 10;
///
///	BaseMatrix<float> mS(1, 1);
///	mS(0, 0) = 10;
///
///	BaseMatrix<float> m3 = m2 - m0;
///	ASSERT_TRUE(m3.IsEqualTo(m1));
///
///	m3 += m0;
///	ASSERT_TRUE(m3.IsEqualTo(m2));
///
///	m3 = m0 + 10;
///	ASSERT_TRUE(m3.IsEqualTo(m1));
///
///	m3 -= 10;
///	ASSERT_TRUE(m3.IsEqualTo(m0));
///
///	m3 = m1 + m0;
///	ASSERT_TRUE(m3.IsEqualTo(m2));
///
///	m3 -= m0;
///	ASSERT_TRUE(m3.IsEqualTo(m1));
///
///	m3 = m1 - 10;
///	ASSERT_TRUE(m3.IsEqualTo(m0));
///
///	m3 += 10;
///	ASSERT_TRUE(m3.IsEqualTo(m1));
///
///	m3 -= mC;
///	ASSERT_TRUE(m3.IsEqualTo(m0));
///
///	m3 += mC;
///	ASSERT_TRUE(m3.IsEqualTo(m1));
///
///	m3 -= mR;
///	ASSERT_TRUE(m3.IsEqualTo(m0));
///
///	m3 += mR;
///	ASSERT_TRUE(m3.IsEqualTo(m1));
///
///	m3.AssignDifferenceOf(m3, mS);
///	ASSERT_TRUE(m3.IsEqualTo(m0));
///}
///
///TEST_F(BaseMatrixTests, BatchMatMul)
///{
///	BaseMatrix<float> m0(6, 2);
///	m0(0, 0) = 1;
///	m0(0, 1) = 2;
///	m0(1, 0) = 3;
///	m0(1, 1) = 4;
///	m0(2, 0) = 5;
///	m0(2, 1) = 6;
///	m0(3, 0) = 7;
///	m0(3, 1) = 8;
///	m0(4, 0) = 9;
///	m0(4, 1) = 10;
///	m0(5, 0) = 11;
///	m0(5, 1) = 12;
///
///	BaseMatrix<float> m1(2, 2);
///	m1(0, 0) = 10;
///	m1(0, 1) = 20;
///	m1(1, 0) = 30;
///	m1(1, 1) = 40;
///
///	BaseMatrix<float> m00(3, 2);
///	m00(0, 0) = 220;
///	m00(0, 1) = 360;
///	m00(1, 0) = 300;
///	m00(1, 1) = 480;
///	m00(2, 0) = 380;
///	m00(2, 1) = 600;
///
///	BaseMatrix<float> m2(3, 2);
///	BaseMatrix<float>::BatchMatMul(0.0, m0, false, 3, m1, false, 1, m2, true);
///	ASSERT_TRUE(m2.IsEqualTo(m00));
///}
///
///TEST_F(BaseMatrixTests, MultiplyAndDiv)
///{
///	BaseMatrix<float> m0(2, 3);
///	m0(0, 0) = 1;
///	m0(0, 1) = 2;
///	m0(0, 2) = 3;
///	m0(1, 0) = 4;
///	m0(1, 1) = 5;
///	m0(1, 2) = 6;
///
///	BaseMatrix<float> m00(2, 3);
///	m00(0, 0) = 10;
///	m00(0, 1) = 20;
///	m00(0, 2) = 30;
///	m00(1, 0) = 40;
///	m00(1, 1) = 50;
///	m00(1, 2) = 60;
///
///	// TODO: consider separate reshape test
///	BaseMatrix<float> m1(2, 3);
///	m1.Reshape(3, 2);
///	m1(0, 0) = 11;
///	m1(0, 1) = 15;
///	m1(1, 0) = 14;
///	m1(1, 1) = 13;
///	m1(2, 0) = 12;
///	m1(2, 1) = 16;
///
///	BaseMatrix<float> m2(2, 2);
///	m2(0, 0) = 75;
///	m2(0, 1) = 89;
///	m2(1, 0) = 186;
///	m2(1, 1) = 221;
///
///	BaseMatrix<float> m3 = m0 * m1;
///	ASSERT_TRUE(m3.IsEqualTo(m2));
///
///	m3 = m0 * 10;
///	ASSERT_TRUE(m3.IsEqualTo(m00));
///
///	m3 = m3 / 10;
///	ASSERT_TRUE(m3.IsEqualTo(m0));
///
///	m3 *= 10;
///	ASSERT_TRUE(m3.IsEqualTo(m00));
///
///	m3 /= 10;
///	ASSERT_TRUE(m3.IsEqualTo(m0));
///
///	BaseMatrix<float>::MultiplyAndWeightedAdd(1, m0, false, m1, false, 0, m3);
///	ASSERT_TRUE(m3.IsEqualTo(m2));
///
///	m1.Reshape(2, 3);
///	BaseMatrix<float>::MultiplyAndWeightedAdd(1, m0, false, m1, true, 0, m3);
///	m2(0, 0) = 74;
///	m2(0, 1) = 92;
///	m2(1, 0) = 182;
///	m2(1, 1) = 227;
///	ASSERT_TRUE(m3.IsEqualTo(m2));
///
///	BaseMatrix<float>::MultiplyAndWeightedAdd(10, m0, false, m1, true, 2, m3);
///	m2(0, 0) = 888;
///	m2(0, 1) = 1104;
///	m2(1, 0) = 2184;
///	m2(1, 1) = 2724;
///	ASSERT_TRUE(m3.IsEqualTo(m2));
///
///	BaseMatrix<float>::MultiplyAndWeightedAdd(1, m0, true, m1, false, 0, m3);
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
///}
///
///TEST_F(BaseMatrixTests, ElementOperations)
///{
///	// TODO: consider splitting this large test
///	BaseMatrix<float> m0(2, 3);
///	m0(0, 0) = 1;
///	m0(0, 1) = 2;
///	m0(0, 2) = 3;
///	m0(1, 0) = 4;
///	m0(1, 1) = 5;
///	m0(1, 2) = 6;
///
///	BaseMatrix<float> m0Inverse(2, 3);
///	m0Inverse(0, 0) = 1.0;
///	m0Inverse(0, 1) = 1 / 2.0;
///	m0Inverse(0, 2) = 1 / 3.0;
///	m0Inverse(1, 0) = 1 / 4.0;
///	m0Inverse(1, 1) = 1 / 5.0;
///	m0Inverse(1, 2) = 1 / 6.0;
///
///	BaseMatrix<float> m1(2, 3);
///	m1(0, 0) = 1;
///	m1(0, 1) = 1;
///	m1(0, 2) = 1;
///	m1(1, 0) = 1;
///	m1(1, 1) = 1;
///	m1(1, 2) = 1;
///
///	BaseMatrix<float> m3;
///	m3.AssignElementProductOf(m0, m0Inverse);
///	ASSERT_TRUE(m3.IsEqualTo(m1, c_epsilonFloatE4));
///
///	m3 = m0 ^ 4;
///	BaseMatrix<float> m2(2, 3);
///	m2(0, 0) = 1;
///	m2(0, 1) = 16;
///	m2(0, 2) = 81;
///	m2(1, 0) = 256;
///	m2(1, 1) = 625;
///	m2(1, 2) = 1296;
///	ASSERT_TRUE(m3.IsEqualTo(m2));
///
///	m3.SetValue(m0);
///	m3 ^= 4;
///	ASSERT_TRUE(m3.IsEqualTo(m2));
///
///	m3.SetValue(m0);
///	m3.ElementMultiplyWith(m0Inverse);
///	ASSERT_TRUE(m3.IsEqualTo(m1));
///
///	m3.SetValue(m0);
///	m3.ElementInverse();
///	ASSERT_TRUE(m3.IsEqualTo(m0Inverse));
///
///	m2(0, 0) = 0.7311;
///	m2(0, 1) = 0.8808;
///	m2(0, 2) = 0.9526;
///	m2(1, 0) = 0.9820;
///	m2(1, 1) = 0.9933;
///	m2(1, 2) = 0.9975;
///	m3.AssignElementDivisionOf(m2, m0);
///	m2.ElementMultiplyWith(m0Inverse);
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));
///
///	m3.SetValue(m0);
///	m3.InplaceSigmoid();
///	m2(0, 0) = 0.7311;
///	m2(0, 1) = 0.8808;
///	m2(0, 2) = 0.9526;
///	m2(1, 0) = 0.9820;
///	m2(1, 1) = 0.9933;
///	m2(1, 2) = 0.9975;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));
///
///	m3.SetValue(m0);
///	m3.InplaceTanh();
///	m2(0, 0) = 0.7616;
///	m2(0, 1) = 0.9640;
///	m2(0, 2) = 0.9951;
///	m2(1, 0) = 0.9993;
///	m2(1, 1) = 0.9999;
///	m2(1, 2) = 1.0000;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));
///
///	m3.InplaceAtanh();
///	ASSERT_TRUE(m3.IsEqualTo(m0, c_epsilonFloatE4));
///
///	m3.SetValue(m0);
///	m3.InplaceLogSoftmax(true);
///	m3.InplaceExp();
///	m2(0, 0) = 0.0474;
///	m2(0, 1) = 0.0474;
///	m2(0, 2) = 0.0474;
///	m2(1, 0) = 0.9526;
///	m2(1, 1) = 0.9526;
///	m2(1, 2) = 0.9526;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));
///
///	m3.SetValue(m0);
///	m3.InplaceLogSoftmax(false);
///	m3.InplaceExp();
///	m2(0, 0) = 0.0900;
///	m2(0, 1) = 0.2447;
///	m2(0, 2) = 0.6652;
///	m2(1, 0) = 0.0900;
///	m2(1, 1) = 0.2447;
///	m2(1, 2) = 0.6652;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));
///
///	m3.SetValue(m0);
///	m3.InplaceHardmax(true);
///	m2(0, 0) = 0.0;
///	m2(0, 1) = 0.0;
///	m2(0, 2) = 0.0;
///	m2(1, 0) = 1.0;
///	m2(1, 1) = 1.0;
///	m2(1, 2) = 1.0;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));
///
///	m3.SetValue(m0);
///	m3.InplaceHardmax(false);
///	m2(0, 0) = 0.0;
///	m2(0, 1) = 0.0;
///	m2(0, 2) = 1.0;
///	m2(1, 0) = 0.0;
///	m2(1, 1) = 0.0;
///	m2(1, 2) = 1.0;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));
///
///	m3.SetValue(m0);
///	m3.InplaceSqrt();
///	m2(0, 0) = 1;
///	m2(0, 1) = 1.4142;
///	m2(0, 2) = 1.7321;
///	m2(1, 0) = 2;
///	m2(1, 1) = 2.2361;
///	m2(1, 2) = 2.4495;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));
///
///	m3.SetValue(m0);
///	m3.InplaceExp();
///	m2(0, 0) = 2.7183;
///	m2(0, 1) = 7.3891;
///	m2(0, 2) = 20.0855;
///	m2(1, 0) = 54.5982;
///	m2(1, 1) = 148.4132;
///	m2(1, 2) = 403.4288;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));
///
///	m3.InplaceLog();
///	ASSERT_TRUE(m3.IsEqualTo(m0, c_epsilonFloatE4));
///
///	m3.SetValue(m0);
///	m3.InplaceTruncateBottom(2);
///	m2(0, 0) = 2;
///	m2(0, 1) = 2;
///	m2(0, 2) = 3;
///	m2(1, 0) = 4;
///	m2(1, 1) = 5;
///	m2(1, 2) = 6;
///	ASSERT_TRUE(m3.IsEqualTo(m2));
///
///	m3.SetValue(m0);
///	m3.InplaceTruncateTop(4);
///	m2(0, 0) = 1;
///	m2(0, 1) = 2;
///	m2(0, 2) = 3;
///	m2(1, 0) = 4;
///	m2(1, 1) = 4;
///	m2(1, 2) = 4;
///	ASSERT_TRUE(m3.IsEqualTo(m2));
///
///	BaseMatrix<float> m_Trig(2, 3);
///	m_Trig(0, 0) = 0;
///	m_Trig(0, 1) = pi / 2.0;
///	m_Trig(0, 2) = pi;
///	m_Trig(1, 0) = 3.0 * pi / 2.0;
///	m_Trig(1, 1) = 2.0 * pi;
///	m_Trig(1, 2) = 5.0 * pi / 2.0;
///
///	BaseMatrix<float> m_Cos(2, 3);
///	m_Cos.SetValue(m_Trig);
///
///	BaseMatrix<float> m_Cos_expected(2, 3);
///	m_Cos_expected(0, 0) = 1;
///	m_Cos_expected(0, 1) = 0;
///	m_Cos_expected(0, 2) = -1;
///	m_Cos_expected(1, 0) = 0;
///	m_Cos_expected(1, 1) = 1;
///	m_Cos_expected(1, 2) = 0;
///
///	m_Cos.InplaceCosine();
///	ASSERT_TRUE(m_Cos.IsEqualTo(m_Cos_expected, c_epsilonFloatE4));
///
///	m_Cos.SetValue(m_Trig);
///	m_Cos.AssignCosineOf(m_Trig);
///	ASSERT_TRUE(m_Cos.IsEqualTo(m_Cos_expected, c_epsilonFloatE4));
///
///	BaseMatrix<float> m_NegSine(2, 3);
///	m_NegSine.SetValue(m_Trig);
///
///	BaseMatrix<float> m_NegSine_expected(2, 3);
///	m_NegSine_expected(0, 0) = 0;
///	m_NegSine_expected(0, 1) = -1;
///	m_NegSine_expected(0, 2) = 0;
///	m_NegSine_expected(1, 0) = 1;
///	m_NegSine_expected(1, 1) = 0;
///	m_NegSine_expected(1, 2) = -1;
///
///	m_NegSine.InplaceNegativeSine();
///	ASSERT_TRUE(m_NegSine.IsEqualTo(m_NegSine_expected, c_epsilonFloatE4));
///
///	m_NegSine.SetValue(m_Trig);
///	m_NegSine.AssignNegativeSineOf(m_Trig);
///	ASSERT_TRUE(m_NegSine.IsEqualTo(m_NegSine_expected, c_epsilonFloatE4));
///
///	m3.SetValue(m0Inverse);
///	m3.InplaceAcos();
///	m2(0, 0) = 0.0;
///	m2(0, 1) = 1.04719755;
///	m2(0, 2) = 1.23095942;
///	m2(1, 0) = 1.31811607;
///	m2(1, 1) = 1.36943841;
///	m2(1, 2) = 1.40334825;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));
///
///	m3.SetValue(m0Inverse);
///	m3.InplaceAsin();
///	m2(0, 0) = 1.57079633;
///	m2(0, 1) = 0.52359878;
///	m2(0, 2) = 0.33983691;
///	m2(1, 0) = 0.25268026;
///	m2(1, 1) = 0.20135792;
///	m2(1, 2) = 0.16744808;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));
///
///	m3.SetValue(m0);
///	m3.InplaceCosh();
///	m2(0, 0) = 1.54308063;
///	m2(0, 1) = 3.76219569;
///	m2(0, 2) = 10.067662;
///	m2(1, 0) = 27.30823284;
///	m2(1, 1) = 74.20994852;
///	m2(1, 2) = 201.71563612;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));
///
///	m3.SetValue(m0);
///	m3.InplaceSinh();
///	m2(0, 0) = 1.17520119;
///	m2(0, 1) = 3.62686041;
///	m2(0, 2) = 10.01787493;
///	m2(1, 0) = 27.2899172;
///	m2(1, 1) = 74.20321058;
///	m2(1, 2) = 201.71315737;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));
///
///	m3.SetValue(m0);
///	m3.InplaceAsinh();
///	m2(0, 0) = 0.88137359;
///	m2(0, 1) = 1.44363548;
///	m2(0, 2) = 1.81844646;
///	m2(1, 0) = 2.09471255;
///	m2(1, 1) = 2.31243834;
///	m2(1, 2) = 2.49177985;
///	ASSERT_TRUE(m3.IsEqualTo(m2, c_epsilonFloatE4));
///}
///
///TEST_F(BaseMatrixTests, Norms)
///{
///	BaseMatrix<float> m0(2, 3);
///	m0(0, 0) = 1;
///	m0(0, 1) = 2;
///	m0(0, 2) = 3;
///	m0(1, 0) = 4;
///	m0(1, 1) = 5;
///	m0(1, 2) = 6;
///
///	BaseMatrix<float> mResult;
///	m0.VectorNorm1(mResult, true);
///	BaseMatrix<float> m2(1, 3);
///	m2(0, 0) = 5;
///	m2(0, 1) = 7;
///	m2(0, 2) = 9;
///	ASSERT_TRUE(mResult.IsEqualTo(m2));
///
///	m0.VectorNorm1(mResult, false);
///	m2.Resize(2, 1);
///	m2(0, 0) = 6;
///	m2(1, 0) = 15;
///	ASSERT_TRUE(mResult.IsEqualTo(m2));
///
///	m0.VectorNorm2(mResult, true);
///	m2.Resize(1, 3);
///	m2(0, 0) = 4.1231;
///	m2(0, 1) = 5.3852;
///	m2(0, 2) = 6.7082;
///	ASSERT_TRUE(mResult.IsEqualTo(m2, c_epsilonFloatE4));
///
///	m0.VectorNorm2(mResult, false);
///	m2.Resize(2, 1);
///	m2(0, 0) = 3.7417;
///	m2(1, 0) = 8.7750;
///	ASSERT_TRUE(mResult.IsEqualTo(m2, c_epsilonFloatE4));
///
///	m0.VectorNormInf(mResult, true);
///	m2.Resize(1, 3);
///	m2(0, 0) = 4;
///	m2(0, 1) = 5;
///	m2(0, 2) = 6;
///	ASSERT_TRUE(mResult.IsEqualTo(m2, c_epsilonFloatE4));
///
///	m0.VectorNormInf(mResult, false);
///	m2.Resize(2, 1);
///	m2(0, 0) = 3;
///	m2(1, 0) = 6;
///	ASSERT_TRUE(mResult.IsEqualTo(m2));
///
///	ASSERT_TRUE(abs(m0.FrobeniusNorm() - 9.5394) < c_epsilonFloatE4);
///	ASSERT_TRUE(abs(m0.MatrixNormInf() - 6) < c_epsilonFloatE4);
///
///	BaseMatrix<float> m1;
///	m0.VectorMax(m1, mResult, true);
///	m2.Resize(1, 3);
///	m2(0, 0) = 4;
///	m2(0, 1) = 5;
///	m2(0, 2) = 6;
///	ASSERT_TRUE(mResult.IsEqualTo(m2, c_epsilonFloatE4));
///
///	m0.VectorMax(m1, mResult, false);
///	m2.Resize(2, 1);
///	m2(0, 0) = 3;
///	m2(1, 0) = 6;
///	ASSERT_TRUE(mResult.IsEqualTo(m2, c_epsilonFloatE4));
///
///	m0.VectorMin(m1, mResult, true);
///	m2.Resize(1, 3);
///	m2(0, 0) = 1;
///	m2(0, 1) = 2;
///	m2(0, 2) = 3;
///	ASSERT_TRUE(mResult.IsEqualTo(m2, c_epsilonFloatE4));
///
///	m0.VectorMin(m1, mResult, false);
///	m2.Resize(2, 1);
///	m2(0, 0) = 1;
///	m2(1, 0) = 4;
///	ASSERT_TRUE(mResult.IsEqualTo(m2, c_epsilonFloatE4));
///}
///
///TEST_F(BaseMatrixTests, SetValues)
///{
///	RandomSeedFixture rsf;
///	BaseMatrix<float> m0(3, 3);
///	m0(0, 0) = 10;
///	m0(1, 1) = 10;
///	m0(2, 2) = 10;
///
///	BaseMatrix<float> m1(3, 3);
///	m1.SetDiagonalValue(10);
///	ASSERT_TRUE(m1.IsEqualTo(m0, c_epsilonFloatE4));
///
///	BaseMatrix<float> m2(3, 1);
///	m2(0, 0) = 10;
///	m2(1, 0) = 10;
///	m2(2, 0) = 10;
///	m1.SetDiagonalValue(m2);
///	ASSERT_TRUE(m1.IsEqualTo(m0, c_epsilonFloatE4));
///
///	m1.SetUniformRandomValue(-0.01, 0.01, rsf.IncrementCounter());
///	foreach_coord (i, j, m1) { ASSERT_TRUE(m1(i, j) >= -0.01 && m1(i, j) < 0.01); }
///
///	m1.Resize(20, 20);
///	m1.SetGaussianRandomValue(1.0, 0.01, rsf.IncrementCounter());
///	ASSERT_NEAR(m1.SumOfElements(), static_cast<double>(m1.GetNumElements()), 1);
///}
///
///TEST_F(BaseMatrixTests, Transpose)
///{
///	BaseMatrix<float> m0(2, 3);
///	m0(0, 0) = 1;
///	m0(0, 1) = 2;
///	m0(0, 2) = 3;
///	m0(1, 0) = 4;
///	m0(1, 1) = 5;
///	m0(1, 2) = 6;
///
///	BaseMatrix<float> m1(3, 2);
///	m1(0, 0) = 1;
///	m1(0, 1) = 4;
///	m1(1, 0) = 2;
///	m1(1, 1) = 5;
///	m1(2, 0) = 3;
///	m1(2, 1) = 6;
///
///	BaseMatrix<float> m2 = m0.Transpose();
///	ASSERT_TRUE(m2.IsEqualTo(m1, c_epsilonFloatE4));
///
///	m2.AssignTransposeOf(m1);
///	ASSERT_TRUE(m2.IsEqualTo(m0, c_epsilonFloatE4));
///}
///
///TEST_F(BaseMatrixTests, ColumnSlice)
///{
///	RandomSeedFixture rsf;
///	BaseMatrix<float> m0(2, 3);
///	m0(0, 0) = 1;
///	m0(0, 1) = 2;
///	m0(0, 2) = 3;
///	m0(1, 0) = 4;
///	m0(1, 1) = 5;
///	m0(1, 2) = 6;
///
///	BaseMatrix<float> m1(2, 2);
///	m1(0, 0) = 1;
///	m1(0, 1) = 2;
///	m1(1, 0) = 4;
///	m1(1, 1) = 5;
///
///	BaseMatrix<float> m2 = m0.GetColumnSlice(0, 2);
///	ASSERT_TRUE(m2.IsEqualTo(m1, c_epsilonFloatE4));
///
///	m1(0, 0) = 2;
///	m1(0, 1) = 3;
///	m1(1, 0) = 5;
///	m1(1, 1) = 6;
///
///	m2 = m0.GetColumnSlice(1, 2);
///	ASSERT_TRUE(m2.IsEqualTo(m1, c_epsilonFloatE4));
///
///	// TODO: this fails due to access violation (at least on desktop machine of pkranen)
///	// size_t k = 100, n = 20, m = 50;
///	// reducing sizes to 2, 20, 5
///	size_t k = 2;
///	size_t n = 20;
///	size_t m = 5;
///
///	BaseMatrix<float> mA(k, n);
///	mA.SetUniformRandomValue(-1, 1, rsf.IncrementCounter());
///
///	BaseMatrix<float> mB(n, m);
///	mB.SetUniformRandomValue(-1, 1, rsf.IncrementCounter());
///
///	BaseMatrix<float> mC(k, m);
///	mC.SetUniformRandomValue(-1, 1, rsf.IncrementCounter());
///
///	BaseMatrix<float> mD(k, m);
///	mD.SetValue(mC);
///
///	BaseMatrix<float>::MultiplyAndAdd(mA, false, mB, false, mD);
///
///	for (int i = 0; i < m; i++)
///	{
///		BaseMatrix<float> colMB = mB.GetColumnSlice(i, 1);
///		BaseMatrix<float> colMC = mC.GetColumnSlice(i, 1);
///		BaseMatrix<float>::MultiplyAndAdd(mA, false, colMB, false, colMC);
///	}
///	ASSERT_TRUE(mC.IsEqualTo(mD, c_epsilonFloatE4));
///}
///
///TEST_F(BaseMatrixTests, CPUKhatriRaoProduct)
///{
///	BaseMatrix<float> mA(3, 4);
///	mA(0, 0) = 0.8147;
///	mA(0, 1) = 0.9134;
///	mA(0, 2) = 0.2785;
///	mA(0, 3) = 0.9649;
///	mA(1, 0) = 0.9058;
///	mA(1, 1) = 0.6324;
///	mA(1, 2) = 0.5469;
///	mA(1, 3) = 0.1576;
///	mA(2, 0) = 0.1270;
///	mA(2, 1) = 0.0975;
///	mA(2, 2) = 0.9575;
///	mA(2, 3) = 0.9706;
///
///	BaseMatrix<float> mB(2, 4);
///	mB(0, 0) = 0.9572;
///	mB(0, 1) = 0.8003;
///	mB(0, 2) = 0.4218;
///	mB(0, 3) = 0.7922;
///	mB(1, 0) = 0.4854;
///	mB(1, 1) = 0.1419;
///	mB(1, 2) = 0.9157;
///	mB(1, 3) = 0.9595;
///
///	BaseMatrix<float> mD(6, 4);
///	mD(0, 0) = 0.7798;
///	mD(0, 1) = 0.7310;
///	mD(0, 2) = 0.1175;
///	mD(0, 3) = 0.7644;
///	mD(1, 0) = 0.8670;
///	mD(1, 1) = 0.5061;
///	mD(1, 2) = 0.2307;
///	mD(1, 3) = 0.1249;
///	mD(2, 0) = 0.1215;
///	mD(2, 1) = 0.0781;
///	mD(2, 2) = 0.4038;
///	mD(2, 3) = 0.7689;
///	mD(3, 0) = 0.3954;
///	mD(3, 1) = 0.1296;
///	mD(3, 2) = 0.2550;
///	mD(3, 3) = 0.9258;
///	mD(4, 0) = 0.4396;
///	mD(4, 1) = 0.0897;
///	mD(4, 2) = 0.5008;
///	mD(4, 3) = 0.1512;
///	mD(5, 0) = 0.0616;
///	mD(5, 1) = 0.0138;
///	mD(5, 2) = 0.8768;
///	mD(5, 3) = 0.9313;
///
///	BaseMatrix<float> mC;
///	mC.AssignKhatriRaoProductOf(mA, mB);
///	ASSERT_TRUE(mC.IsEqualTo(mD, c_epsilonFloatE4));
///}
///
///TEST_F(BaseMatrixTests, CPUAddColumnReshapeProductOf)
///{
///	BaseMatrix<float> mA(6, 2);
///	mA(0, 0) = 0.6557;
///	mA(0, 1) = 0.7431;
///	mA(1, 0) = 0.0357;
///	mA(1, 1) = 0.3922;
///	mA(2, 0) = 0.8491;
///	mA(2, 1) = 0.6555;
///	mA(3, 0) = 0.9340;
///	mA(3, 1) = 0.1712;
///	mA(4, 0) = 0.6787;
///	mA(4, 1) = 0.7060;
///	mA(5, 0) = 0.7577;
///	mA(5, 1) = 0.0318;
///
///	BaseMatrix<float> mB(3, 2);
///	mB(0, 0) = 0.2769;
///	mB(0, 1) = 0.8235;
///	mB(1, 0) = 0.0462;
///	mB(1, 1) = 0.6948;
///	mB(2, 0) = 0.0971;
///	mB(2, 1) = 0.3171;
///
///	BaseMatrix<float> mD(2, 2);
///	mD(0, 0) = 0.2867;
///	mD(0, 1) = 1.2913;
///	mD(1, 0) = 0.1266;
///	mD(1, 1) = 0.4520;
///
///	BaseMatrix<float> mE(2, 2);
///	mE(0, 0) = 0.2657;
///	mE(0, 1) = 1.0923;
///	mE(1, 0) = 0.3636;
///	mE(1, 1) = 0.6416;
///
///	BaseMatrix<float> mC(2, 2);
///	mC.SetValue(0);
///	mC.AddColumnReshapeProductOf(mA, mB, false);
///	ASSERT_TRUE(mC.IsEqualTo(mD, c_epsilonFloatE4));
///
///	mC.SetValue(0);
///	mC.AddColumnReshapeProductOf(mA, mB, true);
///	ASSERT_TRUE(mC.IsEqualTo(mE, c_epsilonFloatE4));
///}
///
///TEST_F(BaseMatrixTests, RowSliceAndStack)
///{
///	BaseMatrix<float> m0(5, 3);
///	m0(0, 0) = 1;
///	m0(0, 1) = 6;
///	m0(0, 2) = 11;
///	m0(1, 0) = 2;
///	m0(1, 1) = 7;
///	m0(1, 2) = 12;
///	m0(2, 0) = 3;
///	m0(2, 1) = 8;
///	m0(2, 2) = 13;
///	m0(3, 0) = 4;
///	m0(3, 1) = 9;
///	m0(3, 2) = 14;
///	m0(4, 0) = 5;
///	m0(4, 1) = 10;
///	m0(4, 2) = 15;
///
///	BaseMatrix<float> m1(2, 3);
///	m1(0, 0) = 3;
///	m1(0, 1) = 8;
///	m1(0, 2) = 13;
///	m1(1, 0) = 4;
///	m1(1, 1) = 9;
///	m1(1, 2) = 14;
///
///	BaseMatrix<float> m2;
///	m2.AssignRowSliceValuesOf(m0, 2, 2);
///	ASSERT_TRUE(m2.IsEqualTo(m1, c_epsilonFloatE4));
///
///	BaseMatrix<float> m3(5, 3);
///	m3(0, 0) = 0;
///	m3(0, 1) = 0;
///	m3(0, 2) = 0;
///	m3(1, 0) = 0;
///	m3(1, 1) = 0;
///	m3(1, 2) = 0;
///	m3(2, 0) = 3;
///	m3(2, 1) = 8;
///	m3(2, 2) = 13;
///	m3(3, 0) = 4;
///	m3(3, 1) = 9;
///	m3(3, 2) = 14;
///	m3(4, 0) = 0;
///	m3(4, 1) = 0;
///	m3(4, 2) = 0;
///
///	m3 += m0;
///	m0.AddToRowSliceValuesOf(m1, 2, 2);
///	ASSERT_TRUE(m3.IsEqualTo(m0, c_epsilonFloatE4));
///
///	m2.AddWithRowSliceValuesOf(m1, 0, 2);
///	BaseMatrix<float> m4(2, 3);
///	m4(0, 0) = 6;
///	m4(0, 1) = 16;
///	m4(0, 2) = 26;
///	m4(1, 0) = 8;
///	m4(1, 1) = 18;
///	m4(1, 2) = 28;
///	ASSERT_TRUE(m2.IsEqualTo(m4, c_epsilonFloatE4));
///}
///
///TEST_F(BaseMatrixTests, CPUAssignRepeatOf)
///{
///	BaseMatrix<float> m0(2, 3);
///	m0(0, 0) = 1;
///	m0(0, 1) = 6;
///	m0(0, 2) = 11;
///	m0(1, 0) = 2;
///	m0(1, 1) = 7;
///	m0(1, 2) = 12;
///
///	BaseMatrix<float> m1;
///	m1.AssignRepeatOf(m0, 1, 1);
///	ASSERT_TRUE(m1.IsEqualTo(m0, c_epsilonFloatE4));
///
///	BaseMatrix<float> m2(6, 6);
///	m2(0, 0) = 1;
///	m2(0, 1) = 6;
///	m2(0, 2) = 11;
///	m2(0, 3) = 1;
///	m2(0, 4) = 6;
///	m2(0, 5) = 11;
///	m2(1, 0) = 2;
///	m2(1, 1) = 7;
///	m2(1, 2) = 12;
///	m2(1, 3) = 2;
///	m2(1, 4) = 7;
///	m2(1, 5) = 12;
///	m2(2, 0) = 1;
///	m2(2, 1) = 6;
///	m2(2, 2) = 11;
///	m2(2, 3) = 1;
///	m2(2, 4) = 6;
///	m2(2, 5) = 11;
///	m2(3, 0) = 2;
///	m2(3, 1) = 7;
///	m2(3, 2) = 12;
///	m2(3, 3) = 2;
///	m2(3, 4) = 7;
///	m2(3, 5) = 12;
///	m2(4, 0) = 1;
///	m2(4, 1) = 6;
///	m2(4, 2) = 11;
///	m2(4, 3) = 1;
///	m2(4, 4) = 6;
///	m2(4, 5) = 11;
///	m2(5, 0) = 2;
///	m2(5, 1) = 7;
///	m2(5, 2) = 12;
///	m2(5, 3) = 2;
///	m2(5, 4) = 7;
///	m2(5, 5) = 12;
///
///	m1.AssignRepeatOf(m0, 3, 2);
///	ASSERT_TRUE(m1.IsEqualTo(m2, c_epsilonFloatE4));
///}
///
///TEST_F(BaseMatrixTests, CPURowElementOperations)
///{
///	RandomSeedFixture rsf;
///	BaseMatrix<float> m0 = BaseMatrix<float>::RandomUniform(20, 28, -1, 1, rsf.IncrementCounter());
///	BaseMatrix<float> m1 = BaseMatrix<float>::RandomUniform(1, 28, 1, 2, rsf.IncrementCounter());
///
///	BaseMatrix<float> m2;
///	m2.SetValue(m0);
///	m2.RowElementMultiplyWith(m1);
///	m2.RowElementDivideBy(m1);
///
///	ASSERT_TRUE(m0.IsEqualTo(m2, c_epsilonFloatE4));
///}
///
///TEST_F(BaseMatrixTests, CPUColumnElementOperations)
///{
///	RandomSeedFixture rsf;
///	BaseMatrix<float> m0 = BaseMatrix<float>::RandomUniform(20, 28, -1, 1, rsf.IncrementCounter());
///	BaseMatrix<float> m1 = BaseMatrix<float>::RandomUniform(20, 1, 1, 2, rsf.IncrementCounter());
///
///	BaseMatrix<float> m2;
///	m2.SetValue(m0);
///	m2.ColumnElementMultiplyWith(m1);
///	m2.ColumnElementDivideBy(m1);
///
///	ASSERT_TRUE(m0.IsEqualTo(m2, c_epsilonFloatE4));
///}
///
///TEST_F(BaseMatrixTests, SeedingFloat)
///{
///	const float low = 0;
///	const float high = 1;
///	const unsigned long seed = 4711;
///
///	auto m1 = BaseMatrix<float>::RandomUniform(16, 16, low, high, seed);
///	auto m2 = BaseMatrix<float>::RandomUniform(16, 16, low, high, seed);
///
///	ASSERT_TRUE(m1.IsEqualTo(m2));
///}
///
///TEST_F(BaseMatrixTests, SeedingDouble)
///{
///	const double low = 0;
///	const double high = 1;
///	const unsigned long seed = 4711;
///
///	auto m1 = BaseMatrix<double>::RandomUniform(16, 16, low, high, seed);
///	auto m2 = BaseMatrix<double>::RandomUniform(16, 16, low, high, seed);
///
///	ASSERT_TRUE(m1.IsEqualTo(m2));
///}
///
///TEST_F(BaseMatrixTests, Adam)
///{
///	BaseMatrix<double> adamMatrix;
///	BaseMatrix<double> gradients(2, 1);
///	BaseMatrix<double> parameters(2, 1);
///	BaseMatrix<double> expectedParameters(2, 1);
///	BaseMatrix<double> expectedStates(2, 2);
///	double gradientValues[] = { 0.1, -0.1 };
///	double paramValues[] = { 0.1, 0.1 };
///	double expectedValues[] = { -0.05811338, 0.25811338 };
///	double expectedStateValues[] = {1e-5, 0.01, 1e-5, -0.01};
///	gradients.SetValue(2, 1, gradientValues, matrixFormatRowMajor);
///	parameters.SetValue(2, 1, paramValues, matrixFormatRowMajor);
///	expectedParameters.SetValue(2, 1, expectedValues, matrixFormatRowMajor);
///	expectedStates.SetValue(2, 2, expectedStateValues, matrixFormatRowMajor);
///	adamMatrix.Adam(gradients, parameters, 0.1, 0.9, 0.999, 0.5, 1e-8, 0.1);
///
///	ASSERT_TRUE(parameters.IsEqualTo(expectedParameters, 1e-6));
///	ASSERT_TRUE(adamMatrix.IsEqualTo(expectedStates, 1e-6));
///
///	double expectedValues2[] = { -0.27059249, 0.47059249 };
///	double expectedStateValues2[] = { 2e-05, 0.019, 2e-05, -0.019 };
///	expectedParameters.SetValue(2, 1, expectedValues2, matrixFormatRowMajor);
///	expectedStates.SetValue(2, 2, expectedStateValues2, matrixFormatRowMajor);
///	adamMatrix.Adam(gradients, parameters, 0.1, 0.9, 0.999, 0.5, 1e-8, 0.1);
///
///	ASSERT_TRUE(parameters.IsEqualTo(expectedParameters, 1e-6));
///	ASSERT_TRUE(adamMatrix.IsEqualTo(expectedStates, 1e-6));
///}
///
///TEST_F(BaseMatrixTests, AdamVarEpsilon)
///{
///	BaseMatrix<double> adamMatrix;
///	BaseMatrix<double> gradients(2, 1);
///	BaseMatrix<double> parameters(2, 1);
///	BaseMatrix<double> expectedParameters(2, 1);
///	BaseMatrix<double> expectedStates(2, 2);
///	double gradientValues[] = { 0.1, -0.1 };
///	double paramValues[] = { 0.1, 0.1 };
///	double expectedValues[] = { 0.0951532672, 0.1048467328 };
///	double expectedStateValues[] = {1e-5, 0.01, 1e-5, -0.01};
///	double epsilon = 0.1;
///
///	gradients.SetValue(2, 1, gradientValues, matrixFormatRowMajor);
///	parameters.SetValue(2, 1, paramValues, matrixFormatRowMajor);
///	expectedParameters.SetValue(2, 1, expectedValues, matrixFormatRowMajor);
///	expectedStates.SetValue(2, 2, expectedStateValues, matrixFormatRowMajor);
///	adamMatrix.Adam(gradients, parameters, 0.1, 0.9, 0.999, 0.5, epsilon, 0.1);
///
///	ASSERT_TRUE(parameters.IsEqualTo(expectedParameters, 1e-6));
///	ASSERT_TRUE(adamMatrix.IsEqualTo(expectedStates, 1e-6));
///
///	double expectedValues2[] = { 0.0860598361, 0.1139401639 };
///	double expectedStateValues2[] = { 2e-05, 0.019, 2e-05, -0.019 };
///	expectedParameters.SetValue(2, 1, expectedValues2, matrixFormatRowMajor);
///	expectedStates.SetValue(2, 2, expectedStateValues2, matrixFormatRowMajor);
///	adamMatrix.Adam(gradients, parameters, 0.1, 0.9, 0.999, 0.5, epsilon, 0.1);
///
///	ASSERT_TRUE(parameters.IsEqualTo(expectedParameters, 1e-6));
///	ASSERT_TRUE(adamMatrix.IsEqualTo(expectedStates, 1e-6));
///}
///
///TEST_F(BaseMatrixTests, OneHot)
///{
///	const size_t num_class = 6;
///
///	BaseMatrix<float> m0(2, 2);
///	m0(0, 0) = 1;
///	m0(0, 1) = 2;
///	m0(1, 0) = 3;
///	m0(1, 1) = 4;
///
///	BaseMatrix<float> expect(12, 2);
///	expect(1, 0) = 1;
///	expect(9, 0) = 1;
///	expect(2, 1) = 1;
///	expect(10, 1) = 1;
///
///	vector<size_t> shape(3);
///	shape[0] = num_class; shape[1] = 2; shape[2] = 2;
///	BaseMatrix<float> m1;
///	m1.AssignOneHot(m0, shape, 0);
///	ASSERT_TRUE(m1.GetNumRows() == 12);
///	ASSERT_TRUE(m1.GetNumCols() == 2);
///	ASSERT_TRUE(m1.IsEqualTo(expect, 1e-6));
///
///	BaseMatrix<float> expect2(12, 2);
///	expect2(2, 0) = 1;
///	expect2(7, 0) = 1;
///	expect2(4, 1) = 1;
///	expect2(9, 1) = 1;
///
///	vector<size_t> shape2(3);
///	shape2[0] = 2; shape2[1] = num_class; shape2[2] = 2;
///	BaseMatrix<float> m2;
///	m2.AssignOneHot(m0, shape2, 1);
///	ASSERT_TRUE(m2.GetNumRows() == 12);
///	ASSERT_TRUE(m2.GetNumCols() == 2);
///	ASSERT_TRUE(m2.IsEqualTo(expect2, 1e-6));
///
///	BaseMatrix<float> dirtyMatrix(2, 2);
///	dirtyMatrix(0, 0) = 1;
///	dirtyMatrix(0, 1) = -1;
///	dirtyMatrix(1, 0) = 7;
///	dirtyMatrix(1, 1) = 4;
///
///	BaseMatrix<float> dirtyExpect(12, 2);
///	dirtyExpect(1, 0) = 1;
///	dirtyExpect(9, 0) = 0;
///	dirtyExpect(2, 1) = 0;
///	dirtyExpect(10, 1) = 1;
///
///	BaseMatrix<float> dirty_m;
///	dirty_m.AssignOneHot(dirtyMatrix, shape, 0);
///	ASSERT_TRUE(dirty_m.GetNumRows() == 12);
///	ASSERT_TRUE(dirty_m.GetNumCols() == 2);
///	ASSERT_TRUE(dirty_m.IsEqualTo(dirtyExpect, 1e-6));
///}
///
///TEST_F(BaseMatrixTests, ScatterToIndices)
///{
///	const size_t row_elements = 2;
///
///	BaseMatrix<float> m0(2, 2);
///	m0(0, 0) = 1;
///	m0(0, 1) = 2;
///	m0(1, 0) = 2;
///	m0(1, 1) = 4;
///
///	BaseMatrix<float> m1(row_elements, 6);
///	m1(0, 1) = m1(1, 1) = 4;
///	m1(0, 2) = m1(1, 2) = 3;
///	m1(0, 3) = m1(1, 3) = 2;
///	m1(0, 4) = m1(1, 4) = 1;
///
///	BaseMatrix<float> m3(4, 2);
///	m3(0, 0) = 1;
///	m3(1, 0) = 2;
///	m3(2, 0) = 3;
///	m3(3, 0) = 4;
///	m3(0, 1) = 5;
///	m3(1, 1) = 6;
///	m3(2, 1) = 7;
///	m3(3, 1) = 8;
///
///	m1.ScatterToIndices(m3, m0, row_elements);
///
///	BaseMatrix<float> expect(row_elements, 6);
///	expect(0, 1) = 5;
///	expect(1, 1) = 6;
///	expect(0, 2) = 11;
///	expect(1, 2) = 13;
///	expect(0, 3) = 2;
///	expect(1, 3) = 2;
///	expect(0, 4) = 8;
///	expect(1, 4) = 9;
///
///	ASSERT_TRUE(m1.IsEqualTo(expect, 1e-6));
///}
///
///TEST_F(BaseMatrixTests, GatherFromTarget)
///{
///	const size_t row_elements = 2;
///
///	BaseMatrix<float> m0(2, 2);
///	m0(0, 0) = 1;
///	m0(0, 1) = 2;
///	m0(1, 0) = 3;
///	m0(1, 1) = 4;
///
///	BaseMatrix<float> m1(row_elements, 6);
///	m1(0, 1) = m1(1, 1) = 4;
///	m1(0, 2) = m1(1, 2) = 3;
///	m1(0, 3) = m1(1, 3) = 2;
///	m1(0, 4) = m1(1, 4) = 1;
///
///	BaseMatrix<float> expect(4, 2);
///	expect(0, 0) = expect(1, 0) = 4;
///	expect(2, 0) = expect(3, 0) = 2;
///	expect(0, 1) = expect(1, 1) = 3;
///	expect(2, 1) = expect(3, 1) = 1;
///
///	BaseMatrix<float> m2;
///	m2.GatherFromTarget(m0, m1, row_elements);
///	ASSERT_TRUE(m2.GetNumRows() == 4);
///	ASSERT_TRUE(m2.GetNumCols() == 2);
///	ASSERT_TRUE(m2.IsEqualTo(expect, 1e-6));
///}

} } } }
