//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include <string>
#include <vector>
#include "Common/File.h"

namespace Microsoft { namespace MSR { namespace CNTK {

struct FileTest
{
	char			m_char;
	wchar_t			m_wchar;
	int				m_int;
	unsigned		m_unsigned;
	long			m_long;
	long long		m_longlong;
	__int64			m_int64;
	float			m_float;
	double			m_double;
	size_t			m_size_t;
	char*			m_str;				// character string, zero terminated
	wchar_t*		m_wstr;				// wide character string, zero terminated
	std::string		m_string;			// std string
	std::wstring	m_wstring;			// std wide string
	std::vector<long> m_vecLong;		// vector of supported type

	std::string		m_err;

	FileTest() : m_str(NULL), m_wstr(NULL) {}
	~FileTest() { delete [] m_str; delete [] m_wstr; }

	void Init();
	bool operator == (const FileTest& ft);

	bool SetError(LPCSTR s) { m_err = s; return false; }
};

File& operator >> (File& ff, FileTest& ft);
File& operator << (File& ff, FileTest& ft);

} } }
