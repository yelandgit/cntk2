//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

//#include <cctype>
//#include <codecvt>
//#include <cwctype>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
//#include <limits.h>

///#if defined(_MSC_VER)
///#include <cuchar>
///
//// These methods aren't in the std namespace on Linux, so make them available in a way that is consistent with
//// both without polluting the global namespace.
//namespace
///{
///	using std::c16rtomb;
///	using std::mbrtoc16;
///	using std::mbrtoc32;
///} // anonymous namespace
///
///#else
//// The versions of GCC that we are using (5.4), don't provide cuchar. However, cuchar is a thin wrapper over
//// uchar.h, so we can include that directly.
///#include <uchar.h>
///
///#endif


std::string unistr(const wchar_t* s, int cp);
inline std::string unistr(const std::wstring& ws, int cp) { return unistr(ws.c_str(),cp); }
std::wstring uniwstr(const char* s, int cp);
inline std::wstring uniwstr(const std::string& ws, int cp) { return uniwstr(ws.c_str(),cp); }

inline std::string utf8c(const wchar_t* s) { return unistr(s, 65001); }
inline std::string utf8c(const std::wstring& ws) { return unistr(ws.c_str(), 65001); }
inline std::wstring utf8wc(const char* s) { return uniwstr(s, 65001); }
inline std::wstring utf8wc(const std::string& s) { return uniwstr(s.c_str(), 65001); }

inline std::string tolower(std::string s) { for (std::string::iterator i=s.begin(); i!=s.end();) *i++ = tolower(*i); return s; }
inline std::string toupper(std::string s) { for (std::string::iterator i=s.begin(); i!=s.end(); ++i) *i = toupper(*i); return s; }
inline std::wstring towlower(std::wstring s) { for (std::wstring::iterator i=s.begin(); i!=s.end();) *i++ = towlower(*i); return s; }
inline std::wstring towupper(std::wstring s) { for (std::wstring::iterator i=s.begin(); i!=s.end(); ++i) *i = towupper(*i); return s; }

inline size_t replace(std::string& s, char c1, char c2) { size_t n=0; for (std::string::iterator i=s.begin(); i!=s.end(); ++i) if (*i==c1) { *i = c2; ++n; } return n; }
inline size_t replace(std::wstring& s, wchar_t c1, wchar_t c2) { size_t n=0; for (std::wstring::iterator i=s.begin(); i!=s.end(); ++i) if (*i==c1) { *i = c2; ++n; } return n; }

inline std::string operator + (const std::string& s1, char c) { std::string s(s1); s += c; return s; }
inline std::string operator + (const std::string& s1, const char* sz) { std::string s(s1); s += sz; return s; }
inline std::string operator + (const std::string& s1, const std::string& s2) { std::string s(s1); s += s2; return s; }
inline std::string operator + (const std::string& s, int d) { char sz[32]; sprintf_s(sz,sizeof(sz),"%d",d); return s + sz; }
inline std::string operator + (const std::string& s, long d) { char sz[32]; sprintf_s(sz,sizeof(sz),"%ld",d); return s + sz; }
inline std::string operator + (const std::string& s, __int64 d) { char sz[32]; sprintf_s(sz,sizeof(sz),"%lld",d); return s + sz; }

inline std::ostream& operator << (std::ostream& os, const std::string& s) { os << s.c_str(); return os; }
inline std::ostream& operator << (std::ostream& os, const wchar_t* s) { if (s[0]) os << utf8c(s).c_str(); return os; }
inline std::ostream& operator << (std::ostream& os, const std::wstring& s) { if (s.size()>0) os << utf8c(s).c_str(); return os; }

typedef std::vector<std::string> strvec;
typedef std::vector<std::wstring> wstrvec;

inline std::ostream& operator << (std::ostream& os, const strvec& v)
{
	for (strvec::const_iterator i=v.begin(); i!=v.end(); ++i)
		os << (*i).c_str() << std::endl;
	return os;
}
inline std::ostream& operator << (std::ostream& os, const wstrvec& v)
{
	for (wstrvec::const_iterator i=v.begin(); i!=v.end(); ++i)
		os << utf8c(*i).c_str() << std::endl;
	return os;
}

size_t tokens(strvec& v, const std::string& s, const char* sep=" \t");

inline std::string hex(short d) { char sz[40]; sprintf_s(sz, sizeof(sz), "%02x", d); return std::string(sz); }
inline std::string hex(unsigned short d) { char sz[40]; sprintf_s(sz, sizeof(sz), "%02x", d); return std::string(sz); }
inline std::string hex(int d) { char sz[40]; sprintf_s(sz, sizeof(sz), "%04x", d); return std::string(sz); }
inline std::string hex(unsigned int d) { char sz[40]; sprintf_s(sz, sizeof(sz), "%04x", d); return std::string(sz); }
inline std::string hex(long d) { char sz[40]; sprintf_s(sz, sizeof(sz), "%08lx", d); return std::string(sz); }
inline std::string hex(unsigned long d) { char sz[40]; sprintf_s(sz, sizeof(sz), "%08lx", d); return std::string(sz); }
inline std::string hex(__int64 d) { char sz[40]; sprintf_s(sz, sizeof(sz), "%016llx", d); return std::string(sz); }
inline std::string hex(unsigned __int64 d) { char sz[40]; sprintf_s(sz, sizeof(sz), "%016llx", d); return std::string(sz); }
inline std::string hex(const void* p) { char sz[40]; sprintf_s(sz, sizeof(sz), "%016llx", (__int64)p); return std::string(sz); }

void xview(std::ostream& os, const void* lp, size_t n, int m=4);

inline void wait(const char* msg="next") { std::cout << msg << "> "; char sz[64]; gets_s(sz,sizeof(sz)); }
