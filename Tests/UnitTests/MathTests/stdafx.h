// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//
#pragma once

#include "targetver.h"
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
#include <windows.h>
#include <stdio.h>
#include <tchar.h>

#ifdef _DEBUG
#include <vld.h>
// citation:
//#ifdef __AFXWIN_H__
//#error[VLD COMPILE ERROR] '#include <vld.h>' should appear before '#include <afxwin.h>' in file stdafx.h
//#endif
#endif

#pragma warning(disable:4996)

#include <array>
//#include <boost/test/unit_test.hpp>
//#include "constants.h"
//#include "fixtures.h"
//#define HALF_IN_BOOST_TEST

#include <iostream>
#include <iomanip>
using namespace std;

//#include "Math/Matrix.h"
//#include "Math/CPUMatrix.h"

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

///void Compare(const Matrix<half>& a, const Matrix<half>& b);

} } } }

