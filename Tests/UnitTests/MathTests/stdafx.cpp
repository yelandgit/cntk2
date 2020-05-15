// stdafx.cpp : source file that includes just the standard includes
// MathTests.pch will be the pre-compiled header
// stdafx.obj will contain the pre-compiled type information

#include "stdafx.h"

// TODO: reference any additional headers you need in STDAFX.H
// and not in this file

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

///void Compare(const Matrix<half>& a, const Matrix<half>& b)
///{
///	cout << endl
///		<< "Compare Matrix<half> " << a.GetNumRows() << "x" << a.GetNumCols() << endl;
///	double dmax = 0;
///	const half* pa = a.Data();
///	const half* pb = b.Data();
///	size_t len = a.GetNumRows()*a.GetNumCols();
///	for (size_t j=0; j<len; ++j)
///	{
///		double d = fabs(float(pa[j]) - float(pb[j])); if (d<=dmax) continue;
///		cout << "\t" << float(pa[j]) << "\t" << float(pb[j]) << "\t" << d << endl;
///		dmax = d;
///	}
///}

} } } }
