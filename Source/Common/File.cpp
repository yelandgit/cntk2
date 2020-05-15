//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// "secure" CRT not available on all platforms
// add this at the top of all CPP files that give "function or variable may be unsafe" warnings
//#ifndef _CRT_SECURE_NO_WARNINGS
//#define _CRT_SECURE_NO_WARNINGS
//#endif
//#define _CRT_NONSTDC_NO_DEPRECATE // make VS accept POSIX functions without _
//
//#define FORMAT_SPECIALIZE // to get the specialized version of the format routines

#include "File.h"
//#include "Config.h"
#include "Directory.h"
//#include <string>
//#include <stdint.h>
//#include <locale>
//#include <unordered_map>
//#ifdef _WIN32
//#define NOMINMAX
//#include <windows.h>
//#ifndef CNTK_UWP
//#include <VersionHelpers.h>
//#endif
//#include <Shlwapi.h>
//#pragma comment(lib, "Shlwapi.lib")
//#endif
//#ifdef __unix__
//#include <unistd.h>
//#include <linux/limits.h> // for PATH_MAX
//#endif

#define FCLOSE_SUCCESS	0
#define PCLOSE_ERROR	-1

#define WRITE_BUFFER_SIZE	(1024 * 1024)

const int CPAGE_UTF8	= 8;
const int CPAGE_UTF16	= 16;

//#include <boost/algorithm/string.hpp>
//#include "../Math/half.hpp"

namespace Microsoft { namespace MSR { namespace CNTK {

static int GetCodePage(FILE* ff)
{
	int cp = CPAGE_UTF8;
	BYTE code[3]; code[2] = 0;
	if (fread(code,1,3,ff)<2) { _fseeki64(ff,0,SEEK_SET); return cp; }
	if (memcmp(code,"\xFF\xFE",2)==0) { cp = CPAGE_UTF16; _fseeki64(ff,2,SEEK_SET); }
	else if (memcmp(code,"\xEF\xBB\xBF",3)!=0) _fseeki64(ff,0,SEEK_SET);
	return cp;
}

bool File::Open(LPCWSTR file, int opts)
{
	if (m_file) Close();
	m_fname = file;
	m_options = 0;
	m_pipe = false;
	m_offset.clear();
	m_input.clear();
	m_item = m_cpage = 0;
	if (m_fname.empty()) return Error("empty file name");

	bool inpipe = (m_fname.back()  == '|');
	bool outpipe = (m_fname.front() == '|');

	wstring options;
	if (opts & FILE_READ) options = L"r";
	else if (opts & FILE_WRITE) options = L"w";
	else if (opts & FILE_APPEND) options = L"a";
	else return Error("Invalid file access mode");
	if (opts & FILE_MODIFY) options.push_back('+');
	//if (opts & FILE_BINARY) options.push_back('b');
	options.push_back('b');

	if (opts & FILE_MODIFY) m_options = FILE_READ | FILE_WRITE;
	else if (opts & FILE_READ) m_options = FILE_READ;
	else m_options = FILE_WRITE;
	m_options |= opts & FILE_BINARY;

///	// convert options to fopen()'s mode string
///	wstring options(reading ? L"r" : L"");
///	if (writing || appending)
///	{
///		options.push_back(writing ? 'w' : 'a');
///		if (appending) m_options |= FILE_WRITE;
///		if (!outpipe && m_fname!=L"-")
///		{
///			options.push_back('+');
///			CreateDir(m_fname.c_str());
///		}
///	}
///	options.push_back(opts & FILE_BINARY ? 'b' : 't');
///	//if (opts & F_SEQ) options += "S";

	if (m_fname == L"-")
	{
		// stdin/stdout
///		if (writing && reading) return Error("Cannot read/write at once with stdin/stdout");
///		m_file = writing ? stdout : stdin;
///		m_cpage = CPAGE_UTF8;
		return Error("stdio/stdout is not supported");
	}
	else if (outpipe || inpipe)
	{
		// pipe syntax: "|cmd" or "cmd|"
///#ifdef CNTK_UWP
		return Error("Pipes are not supported");
///#else
///		if (inputPipe && outputPipe) return Error("Pipes cannot specify read/write at once");
///		if (inputPipe != reading) return Error("Pipes must use consistent read/write");
///		
///		string fname = inputPipe ? m_fname.substr(0, m_fname.size()-1) : m_fname.substr(1);
///		m_file = _wpopen(utf8wc(fname).c_str(), utf8wc(options).c_str());
///		if (m_file==nullptr) return Error("Open pipe; ", errno);
///		m_pipe = true;
///#endif
	}
	else
	{
		// create directory
		if ((opts & (FILE_WRITE | FILE_APPEND))!=0 && !CreateDirs(m_fname))
			return Error("Cannot create directory");

		// open file
		//cout << "  " << utf8c(options) << "\t" << file << endl;
		errno_t res = _wfopen_s(&m_file, file, options.c_str());
		if (res) return Error("Open file; ", res);
		m_cpage = IsTextReading() ? GetCodePage(m_file) : CPAGE_UTF8;
	}
	return true;
}

bool File::Close()
{
	if (m_file==nullptr) return true;
	if (m_pipe)
	{
#ifdef CNTK_UWP
		assert(false); // cannot happen
#else
		int rc = _pclose(m_file);
		if (rc==PCLOSE_ERROR && !uncaught_exception()) return Error("Pipe close error");
#endif
	}
	else if (m_file != stdin && m_file != stdout && m_file != stderr)
	{
		int rc = fclose(m_file);
		if (rc!=FCLOSE_SUCCESS && !uncaught_exception()) return Error("File close error");
	}
	m_options = 0;
	m_pipe = false;
	m_file = nullptr;
	m_err.clear();
	return true;
}

bool File::ReadText(vector<string>& text, LPCWSTR file)
{
	File ff;
	if (!ff.Open(file, FILE_READ|FILE_TEXT)) return false;
	size_t fsize = ff.Size(); text.clear();
	if (fsize>0) { text.reserve(fsize/20 + 1); for (string s; ff.GetLine(s); text.push_back(s)); }
	return true;
}

bool File::ReadText(vector<wstring>& text, LPCWSTR file)
{
	File ff;
	if (!ff.Open(file, FILE_READ|FILE_TEXT)) return false;
	size_t fsize = ff.Size(); text.clear();
	if (fsize>0) { text.reserve(fsize/20 + 1); for (wstring s; ff.GetLine(s); text.push_back(s)); }
	return true;
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//			position
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

size_t File::Size()
{
	if (m_file==nullptr || m_pipe) return 0;
	size_t cpos = _ftelli64(m_file); _fseeki64(m_file, 0, SEEK_END);
	size_t len = _ftelli64(m_file); _fseeki64(m_file, cpos, SEEK_SET);
	return len;
}

size_t File::GetPosition()
{
	return (m_file==nullptr || m_pipe) ? 0 : _ftelli64(m_file);
}

size_t File::SetPosition(size_t pos)
{
	if (m_file==nullptr || m_pipe) return 0;
	size_t cpos = _ftelli64(m_file); _fseeki64(m_file, pos, SEEK_SET);
	if (pos==0) m_cpage = IsTextReading() ? GetCodePage(m_file) : CPAGE_UTF8;
	return cpos;
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//			binary format
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

int File::GetChar()
{
	if (!IsOpenRead()) return (m_cpage==CPAGE_UTF16 ? WEOF : EOF);
	return (m_cpage==CPAGE_UTF16 ? fgetwc(m_file) : fgetc(m_file));
}

bool File::GetData(void* p, size_t n)
{
	return (IsOpenRead() ? fread(p,1,n,m_file)==n : false);
}

bool File::PutData(const void* p, size_t n)
{
	return (IsOpenWrite() ? fwrite(p,1,n,m_file)==n : false);
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//			common I/O
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

bool File::Get(char& c)
{
	if (IsText()) return GetItem(c);
	if (!IsOpenRead()) return false;
	c = fgetc(m_file);
	return true;
}

bool File::Get(wchar_t& wc)
{
	if (IsText()) return GetItem(wc);
	if (!IsOpenRead()) return false;
	wc = fgetwc(m_file);
	return true;
}

// binary format
inline string getstr(FILE* ff)
{
	string s;
	for (char c;;)
		if ((c=fgetc(ff))==0 || c==EOF) break;
		else s.push_back(c);
	return s;
}

bool File::Get(LPSTR sz, size_t n)
{
	if (IsText()) return GetItem(sz,n);
	if (!IsOpenRead()) return false;
	string s = getstr(m_file);
	if (s.size()>=n) return Error("small buffer");
	memcpy(sz, s.c_str(), s.size()+1);
	return true;
}

bool File::Get(LPWSTR wsz, size_t n)
{
	if (IsText()) return GetItem(wsz,n);
	if (!IsOpenRead()) return false;
	wstring ws = utf8wc(getstr(m_file));
	if (ws.size()>=n) return Error("small buffer");
	memcpy(wsz, ws.c_str(), (ws.size()+1)*sizeof(wchar_t));
	return true;
}

bool File::Get(string& s)
{
	if (IsText()) return GetItem(s);
	s = getstr(m_file);
	return true;
}

bool File::Get(wstring& ws)
{
	if (IsText()) return GetItem(ws);
	ws = utf8wc(getstr(m_file));
	return true;
}

bool File::Get(bool& f)
{
	if (IsText()) return GetItem(f);
	if (!IsOpenRead()) return false;
	f = (fgetc(m_file)!=0);
	return true;
}

bool File::Put(char c)
{
	if (IsText()) return GetItem(c);
	if (IsOpenWrite()) { fputc(c, m_file); return true; }
	return false;
}

bool File::Put(wchar_t wc)
{
	if (IsText()) return GetItem(wc);
	if (IsOpenWrite()) { fputwc(wc, m_file); return true; }
	return false;
}

bool File::Put(const string& s)
{
	if (IsText()) return PutItem(s);
	if (!IsOpenWrite()) return false;
	size_t n = s.size() + 1;
	return fwrite(s.c_str(),1,n,m_file)==n;
}

bool File::Put(bool f)
{
	if (IsText()) return PutItem(f);
	if (IsOpenWrite()) { fputc(char(f),m_file); return true; }
	return false;
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//			text stream
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

bool File::PutItem(char c)
{
	if (m_file==nullptr) return false;
	if (c=='\n') { fputc(c, m_file); m_item = 0; }
	else if (++m_item==1) fprintf(m_file, "%s%c", m_offset.c_str(), c);
	else fprintf(m_file, " %c", c);
	return true;
}

bool File::PutItem(wchar_t wc)
{
	if (wc<=127) return PutItem(char(wc));
	wchar_t wsz[2]; wsz[0] = wc; wsz[1] = 0;
	string s = utf8c(wsz); if (++m_item==1) fputc(' ', m_file);
	for (string::iterator i=s.begin(); i!=s.end(); ++i) fputc(*i, m_file);
	return true;
}

bool File::PutItem(LPCSTR s)
{
	if (m_file==nullptr) return false;
	if (++m_item==1) fprintf(m_file, "%s%s", m_offset.c_str(), s);
	else fprintf(m_file, " %s", s);
	return true;
}

string File::GetItem()
{
	if (!IsOpenRead()) return "";

	string s;
	if (m_cpage==CPAGE_UTF16)
	{
		wstring ws;
		for (int ns=0; feof(m_file)==0; )
		{
			wint_t wc = fgetwc(m_file);
			if (wc==WEOF || wc==0) break;
			if (iswspace(wc))
			{
				if (ws.size()==0) continue;
				if (ns==0 || wc=='\n') break;
			}
			if (wc=='"') ns = ws.empty() ? 1:0;
			ws.push_back(wc);
		}
		if (ws.size()>0) s = utf8c(ws);
	}
	else
	{
		for (int ns=0;;)
		{
			char c = fgetc(m_file);
			if (c==EOF || c==0) break;
			if (c==' ' || c=='\n' || c=='\t')
			{
				if (s.size()==0) continue;
				if (ns==0 || c=='\n') break;
			}
			if (c=='"') ns = s.empty() ? 1:0;
			s.push_back(c);
		}
	}
	while (s.size()>0 && s.back()=='\r') s.pop_back();
	return s;
}

bool File::GetItem(LPSTR sz, size_t n)
{
	size_t cpos = GetPosition(); m_err.clear();
	string s = GetItem(); if (s.empty()) return false;
	if (s.size()>=n) { _fseeki64(m_file, cpos, SEEK_SET); return Error("small buffer"); }
	memcpy(sz, s.c_str(), s.size()+1);
	return true;
}

bool File::GetItem(LPWSTR wsz, size_t n)
{
	size_t cpos = GetPosition(); m_err.clear();
	string s = GetItem(); if (s.empty()) return false;
	wstring ws = utf8wc(s);
	if (ws.size()>=n) { _fseeki64(m_file, cpos, SEEK_SET); return Error("small buffer"); }
	memcpy(wsz, ws.c_str(), (ws.size()+1)*sizeof(wchar_t));
	return true;
}

bool File::GetItem(bool& f)
{
	string s = GetItem(); if (s.empty()) return false;
	if (toupper(s)=="T" || s=="TRUE") { f = true; return true; }
	if (s=="F" || s=="FALSE") { f = false; return true; }
	return false;
}

bool File::PutMarkerBegin(LPCSTR s)
{
	if (m_file==nullptr) return false;
	fprintf(m_file, (m_item ? "\n%s%s\n":"%s%s\n"), m_offset.c_str(), s);
	m_offset += "  "; m_item = 0;
	return true;
}

bool File::PutMarkerEnd(LPCSTR s)
{
	if (!IsOpenWrite()) return false;
	if (m_item>0) fprintf(m_file, "\n");
	if (m_offset.size()>1) m_offset.resize(m_offset.size()-2);
	fprintf(m_file, "%s%s\n", m_offset.c_str(), s);
	m_item = 0;
	return true;
}

bool File::GetMarker(LPCSTR s)
{
	if (!IsOpenRead()) return false;
	return GetItem().compare(s)==0;
}

bool File::TryGetMarker(LPCSTR s)
{
	if (!IsOpenRead()) return false;
	size_t pos = GetPosition(); if (GetMarker(s)) return true;
	_fseeki64(m_file, pos, SEEK_SET);
	return false;
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//			...
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

bool File::GetLine(string& s)
{
	if (!IsReading()) return false;
	if (m_cpage==CPAGE_UTF8)
	{
		for (s.clear();;)
		{
			char c = fgetc(m_file);
			if (c=='\n' || c==0 || c==EOF) break;
			s.push_back(c);
		}
		if (s.empty()) return !IsEOF();
		if (s.back()=='\r') s.pop_back();
		return true;
	}
	wstring ws;
	if (GetLine(ws)) { s = utf8c(ws); return true; }
	return false;
}

bool File::GetLine(wstring& ws)
{
	if (!IsReading()) return false;
	if (m_cpage==CPAGE_UTF16)
	{
		for (ws.clear();;)
		{
			wchar_t wc = fgetwc(m_file);
			if (wc=='\n' || wc==0 || wc==WEOF) break;
			ws.push_back(wc);
		}
		if (ws.empty()) return !IsEOF();
		if (ws.back()=='\r') ws.pop_back();
		return true;
	}
	string s;
	if (GetLine(s)) { ws = utf8wc(s); return true; }
	return false;
}

} } }
