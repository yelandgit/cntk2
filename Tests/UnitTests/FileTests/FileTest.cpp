//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// FileTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
///#include "Basics.h"
///#include "FileUtils.h"
#include "FileTest.h"
///#include "File.h"
///#include "Matrix.h"

using namespace Microsoft::MSR::CNTK;

///void MatrixFileWriteAndRead()
///{
///	CPUMatrix<float> M = CPUMatrix<float>::RandomUniform(43, 10, -26.3f, 30.2f);
///	CPUMatrix<float> Mcopy(M);
///	std::wstring filename(L"c:\\temp\\M.txt");
///	File file(filename, fileOptionsUnicode | fileOptionsReadWrite);
///	file << M;
///	CPUMatrix<float> M1;
///	file.SetPosition(0);
///	file >> M1;
///	if (!Mcopy.IsEqualTo(M1))
///		fprintf(stderr, "matrix read/write doesn't pass");
///}
///
///	if (!(options & fileOptionsType))
///	{
///		options |= fileOptionsText;
///		fprintf(stderr, "No file type specified, using UTF-8 text\n");
///	}
///	if (!(options & fileOptionsReadWrite))
///	{
///		options |= fileOptionsReadWrite;
///		fprintf(stderr, "No read or write specified, using read/write\n");
///	}
///	const wchar_t* filename = NULL;
///	for (const wchar_t* arg = args.shift(); arg; arg = args.shift())
///	{
///		filename = arg;
///	}
///	if (filename == NULL)
///	{
///		fprintf(stderr, "filename expected after options\n");
///		goto exit;
///	}
///	TestFileAPI(filename, options);
///exit:
///	return 0;
///}

namespace Microsoft { namespace MSR { namespace CNTK {

void FileTest::Init()
{
	delete [] m_str; m_str = NULL;
	delete [] m_wstr; m_wstr = NULL;
	m_string.clear();
	m_wstring.clear();
	m_vecLong.clear();
	m_err.clear();

	m_char = 'c';
	m_wchar = L'W';
	m_int = 123456;
	m_unsigned = 0xfbadf00d;
	m_long = 0xace;
	m_longlong = 0xaddedbadbeef;
	m_int64 = 0xfeedfacef00d;
	m_size_t = 0xbadfadfacade;
	m_float = 1.23456789e-012f;
	m_double = 9.8765432109876548e-098;

	m_str = new char[80];
	strcpy_s(m_str, 80, "sampleString");			// character string, zero terminated
	m_wstr = new wchar_t[80];
	wcscpy_s(m_wstr, 80, L"wideSampleString");		// wide character string, zero terminated

	m_string.append("std:stringSampleString");		// std string
	m_wstring.append(L"std:wstringSampleString");	// std wide string
	m_vecLong.push_back(m_int);						// vector of supported type
	m_vecLong.push_back(m_unsigned);
	m_vecLong.push_back(m_long);
}

bool FileTest::operator == (const FileTest& ft)
{
	if (m_char!=ft.m_char) return SetError("error; char");
	if (m_wchar!=ft.m_wchar) return SetError("error; wchar");
	if (m_int!=ft.m_int) return SetError("error; m_int");
	if (m_unsigned!=ft.m_unsigned) return SetError("error; m_unsigned");
	if (m_long!=ft.m_long) return SetError("error; m_long");
	if (m_longlong!=ft.m_longlong) return SetError("error; m_longlong");
	if (m_int64!=ft.m_int64) return SetError("error; m_int64");
	if (m_size_t!=ft.m_size_t) return SetError("error; m_size_t");
	if (m_float!=ft.m_float) return SetError("error; m_float");
	if (m_double!=ft.m_double) return SetError("error; m_double");

	int m = (m_str==NULL ? 0:1) + (m_wstr==NULL ? 0:2);
	if (m==1 || m==2 ) return SetError("error; m_str=null");
	if (m==3 && strcmp(m_str,ft.m_str)!=0) return SetError("error; m_str");

	m = (m_wstr==NULL ? 0:1) + (m_wstr==NULL ? 0:2);
	if (m==1 || m==2 ) return SetError("error; m_wstr=null");
	if (m==3 && wcscmp(m_wstr,ft.m_wstr)!=0) return SetError("error; m_wstr");

	if (m_string!=ft.m_string) return SetError("error; m_string");
	if (m_wstring!=ft.m_wstring) return SetError("error; m_wstring");

///	if (m_vecLong.size()!=ft.m_vecLong.size()) return SetError("error; m_vecLong.size()");
///	for (size_t j=0; j<m_vecLong.size(); ++j)
///		if (m_vecLong[j]!=ft.m_vecLong[j]) return SetError("error; m_vecLong");

	return true;
}

File& operator << (File& ff, FileTest& ft)
{
	ff.PutMarkerBegin("BeginFileTest");
	ff << ft.m_char << ft.m_wchar << eoln;
	ff << ft.m_int << ft.m_unsigned << ft.m_long << eoln;
	ff << ft.m_longlong << ft.m_int64 << ft.m_size_t << eoln;
	ff << ft.m_float << ft.m_double << eoln;
	ff << ft.m_str << ft.m_wstr << eoln;
	ff << ft.m_string << ft.m_wstring << eoln;
///	ff << ft.m_vecLong;
	ff.PutMarkerEnd("EndFileTest");
	return ff;
}

File& operator >> (File& ff, FileTest& ft)
{
	if (!ff.GetMarker("BeginFileTest")) { ft.SetError("BeginFileTest not found"); return ff; }
	ff >> ft.m_char >> ft.m_wchar;
	ff >> ft.m_int >> ft.m_unsigned >> ft.m_long;
	ff >> ft.m_longlong;
	ff >> ft.m_int64;
	ff >> ft.m_size_t;
	ff >> ft.m_float >> ft.m_double;
	if (ft.m_str==NULL) ft.m_str = new char[80];
	ff.GetItem(ft.m_str, 80);
	if (ft.m_wstr==NULL) ft.m_wstr = new wchar_t[80];
	ff.GetItem(ft.m_wstr, 80);
	ff >> ft.m_string >> ft.m_wstring;
///	ff >> ft.m_vecLong;
	if (!ff.GetMarker("EndFileTest")) { ft.SetError("EndFileTest not found"); return ff; }
	return ff;
}

} } }
