//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "common.h"

#ifdef CPUONLY
const int c_deviceIdZero = CPUDEVICE;
#else
const int c_deviceIdZero = 0;
#endif

const float  c_epsilonFloatE1	= 0.1f;
const float  c_epsilonFloatE2	= 0.01f;
const float  c_epsilonFloatE3	= 0.001f;
const float  c_epsilonFloatE4	= 0.0001f;
const float  c_epsilonFloat3E4	= 0.0003f;
const float  c_epsilonFloat5E4	= 0.0005f;
const float  c_epsilonFloatE5	= 1e-5f;
const float  c_epsilonFloatE6	= 1e-6f;
const float  c_epsilonFloatE7	= 1e-7f;
const float  c_epsilonFloatE8	= 1e-8f;
const double c_epsilonDoubleE11	= 1e-11;

//template <>
//const float Microsoft::MSR::CNTK::Test::Err<float>::Rel = 1e-5f;
//template <>
//const double Microsoft::MSR::CNTK::Test::Err<double>::Rel = 1e-5f;
//template <>
//const float Microsoft::MSR::CNTK::Test::Err<float>::Abs = 1.192092896e-07f;
//template <>
//const double Microsoft::MSR::CNTK::Test::Err<double>::Abs = 2.2204460492503131e-016;


void Random(SparseData<float>& v, size_t rows, size_t cols, size_t n)
{
	v.clear();
	if (n==0)
	{
		// whole matrix
		v.reserve(rows*cols);
		for (size_t j=0; j<cols; ++j)
		for (size_t i=0; i<rows; ++i)
			v.push_back(ElemItem<float>(i, j, float(0.01*(rand() % 100))));
		return;
	}
	v.reserve(n<rows*cols ? n : n=rows*cols);
	while (v.size()<n)
	{
		while (v.size()<n)
		{
			float val = float(0.01*(rand() % 100)); if (val==0) continue;
			size_t r = rand() % rows, c = rand() % cols;
			v.push_back(ElemItem<float>(r,c,val));
		}
		v.SortByCols();
		ElemItem<float> pi(-1,-1);
		for (SparseData<float>::iterator i=v.begin(); i!=v.end(); )
			if ((*i).row==pi.row && (*i).col==pi.col) i = v.erase(i);
			else pi = *i++;
	}
}

void Random(BaseMatrix<float>& m, size_t n)
{
	SparseData<float> spd;
	Random(spd, m.GetNumRows(), m.GetNumCols(), n);
	m.PutSparseData(spd);
}

