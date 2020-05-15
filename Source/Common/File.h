//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
//#include <stdio.h>
//#include <string>
//#include <vector>
//#include <stdint.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif// NOMINMAX
#include <windows.h>
#endif
#ifdef __unix__
#include <unistd.h>
#endif
//#include "FileUtils.h"// for f{ge,pu}t{,Text}()
#include <fstream>// for LoadMatrixFromTextFile() --TODO: change to using this File class
#include <sstream>

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

#ifndef eoln
#define eoln '\n'
#endif

#define FILE_READ		0x0001		// r
#define FILE_WRITE		0x0002		// w
#define FILE_APPEND		0x0004		// a
#define FILE_MODIFY		0x0008		// +
#define FILE_TEXT		0x0000		// t
#define FILE_BINARY		0x0010		// b


class File
{
private:
	wstring		m_fname;
	string		m_offset;
	string		m_input;
	string		m_err;
	FILE*		m_file;
	DWORD		m_options;
	DWORD		m_cpage;			// UTF-8/16
	int			m_item;				// line item counter
	bool		m_pipe;				// pipe

public:
	File() : m_file(nullptr), m_options(0), m_cpage(0) {}
	File(const wchar_t* file, int opts) { Open(file, opts); }
	File(const wstring&  file, int opts) { Open(file.c_str(), opts); }
	~File() { Close(); }

	bool Open(LPCWSTR file, int opts);
	bool Open(const wstring& file, int opts) { return Open(file.c_str(), opts); }
	bool Flush() { return IsOpen() && fflush(m_file)==0; }
	bool Close();

	LPCWSTR Name() const { return m_fname.c_str(); }

	bool IsOpen() const { return m_file!=nullptr; }
	bool IsOpenRead() const { return m_file!=nullptr && IsReading(); }
	bool IsOpenWrite() const { return m_file!=nullptr && IsWriting(); }
	bool IsPipe() const { return m_pipe; }
	bool IsFile() const { return !m_pipe; }
	bool IsText() const { return (m_options & FILE_BINARY)==0; }
	bool IsBinary() const { return (m_options & FILE_BINARY)!=0; }
	//bool IsTextFile() const { return (m_options & FILE_BINARY)==0 && !m_pipe; }
	bool IsReading() const { return (m_options & FILE_READ)!=0; }
	bool IsWriting() const { return (m_options & FILE_WRITE)!=0; }
	bool IsTextReading() const { return (m_options & (FILE_BINARY+FILE_READ))==FILE_READ; }
	bool IsBinaryReading() const { return (m_options & (FILE_BINARY+FILE_READ))==FILE_BINARY+FILE_READ; }
	bool IsTextWriting() const { return (m_options & (FILE_BINARY+FILE_WRITE))==FILE_WRITE; }
	bool IsBinaryWriting() const { return (m_options & (FILE_BINARY+FILE_WRITE))==FILE_BINARY+FILE_WRITE; }
	bool CanSeek() const { return !m_pipe; }

	bool IsEOF() const { return m_file==nullptr ? true : feof(m_file)!=0; }

	bool IsError() const { return m_err.size()>0; }
	const char* GetError() const { return m_err.c_str(); }

	int  CodePage() const { return m_cpage; }

	size_t Size();
	size_t GetPosition();
	size_t SetPosition(size_t pos);

	// binary format
	int  GetChar();
	bool GetData(void* p, size_t n);
	bool PutData(const void* p, size_t n);

	// common I/O
	bool Get(bool& f);
	bool Get(char& c);
	bool Get(wchar_t& wc);
	bool Get(short& d) { if (IsText()) return GetItem(d); return (IsOpenRead() ? fread(&d,1,sizeof(d),m_file) : false); }
	bool Get(int& d) { if (IsText()) return GetItem(d); return (IsOpenRead() ? fread(&d,1,sizeof(d),m_file) : false); }
	bool Get(long& d) { if (IsText()) return GetItem(d); return (IsOpenRead() ? fread(&d,1,sizeof(d),m_file) : false); }
	bool Get(unsigned short& d) { if (IsText()) return GetItem(d); return (IsOpenRead() ? fread(&d,1,sizeof(d),m_file) : false); }
	bool Get(unsigned int& d) { if (IsText()) return GetItem(d); return (IsOpenRead() ? fread(&d,1,sizeof(d),m_file) : false); }
	bool Get(unsigned long& d) { if (IsText()) return GetItem(d); return (IsOpenRead() ? fread(&d,1,sizeof(d),m_file) : false); }
	bool Get(float& d) { if (IsText()) return GetItem(d); return (IsOpenRead() ? fread(&d,1,sizeof(d),m_file) : false); }
	bool Get(double& d) { if (IsText()) return GetItem(d); return (IsOpenRead() ? fread(&d,1,sizeof(d),m_file) : false); }
	bool Get(long long& d) { if (IsText()) return GetItem(d); return (IsOpenRead() ? fread(&d,1,sizeof(d),m_file) : false); }
	bool Get(unsigned long long& d) { if (IsText()) return GetItem(d); return (IsOpenRead() ? fread(&d,1,sizeof(d),m_file) : false); }
	bool Get(LPSTR s, size_t n);
	bool Get(LPWSTR ws, size_t n);
	bool Get(string& s);
	bool Get(wstring& ws);

	template<class T>
	bool Get(vector<T>& v)
	{
		if (IsText()) return GetItem(v);
		if (!IsOpenRead()) return false;
		DWORD n; if (fread(&n,sizeof(n),1,m_file)!=1) return false;
		v.clear(); v.resize(n);
		return fread(&v[0], sizeof(T), n, m_file)==n;
	}

	bool Put(bool f);
	bool Put(char c);
	bool Put(wchar_t c);
	bool Put(short d) { if (IsText()) return PutItem(d); return IsOpenWrite() ? fwrite(&d,sizeof(d),1,m_file)==1 : false; }
	bool Put(int d) { if (IsText()) return PutItem(d); return IsOpenWrite() ? fwrite(&d,sizeof(d),1,m_file)==1 : false; }
	bool Put(long d) { if (IsText()) return PutItem(d); return IsOpenWrite() ? fwrite(&d,sizeof(d),1,m_file)==1 : false; }
	bool Put(unsigned short d) { if (IsText()) return PutItem(d); return IsOpenWrite() ? fwrite(&d,sizeof(d),1,m_file)==1 : false; }
	bool Put(unsigned int d) { if (IsText()) return PutItem(d); return IsOpenWrite() ? fwrite(&d,sizeof(d),1,m_file)==1 : false; }
	bool Put(unsigned long d) { if (IsText()) return PutItem(d); return IsOpenWrite() ? fwrite(&d,sizeof(d),1,m_file)==1 : false; }
	bool Put(float d) { if (IsText()) return PutItem(d); return IsOpenWrite() ? fwrite(&d,sizeof(d),1,m_file)==1 : false; }
	bool Put(double d) { if (IsText()) return PutItem(d); return IsOpenWrite() ? fwrite(&d,sizeof(d),1,m_file)==1 : false; }
	bool Put(long long d) { if (IsText()) return PutItem(d); return IsOpenWrite() ? fwrite(&d,sizeof(d),1,m_file)==1 : false; }
	bool Put(unsigned long long d) { if (IsText()) return PutItem(d); return IsOpenWrite() ? fwrite(&d,sizeof(d),1,m_file)==1 : false; }
	bool Put(LPCSTR sz) { return Put(string(sz)); }
	bool Put(LPCWSTR wsz) { return Put(utf8c(wsz)); }
	bool Put(const string& s);
	bool Put(const wstring& ws) { return Put(utf8c(ws)); }

	template<class T>
	bool Put(vector<T>& v)
	{
		if (IsText()) return PutItem(v);
		if (!IsOpenWrite()) return false;
		DWORD n = DWORD(v.size());
		if (fwrite(&n,sizeof(n),1,m_file)!=1) return false;
		return fwrite(&v[0],sizeof(T)v.size(),m_file)==v.size();
	}

	// text stream
	string GetItem();

	bool GetItem(bool& f);
	bool GetItem(char& c) { c = ' '; while (isspace(c&0x7f)) c = char(GetChar()); return true; }
	bool GetItem(wchar_t& wc) { wc = ' '; while (iswspace(wc)) wc = GetChar(); return true; }
	bool GetItem(short& d) { string s=GetItem(); if (s.empty()) return false; sscanf_s(s.c_str(),"%hi",&d); return true; }
	bool GetItem(int& d) { string s=GetItem(); if (s.empty()) return false; sscanf_s(s.c_str(),"%i",&d); return true; }
	bool GetItem(long& d) { string s=GetItem(); if (s.empty()) return false; sscanf_s(s.c_str(),"%li",&d); return true; }
	bool GetItem(unsigned short& d) { string s=GetItem(); if (s.empty()) return false; sscanf_s(s.c_str(),"%hu",&d); return true; }
	bool GetItem(unsigned int& d) { string s=GetItem(); if (s.empty()) return false; sscanf_s(s.c_str(),"%u",&d); return true; }
	bool GetItem(unsigned long& d) { string s=GetItem(); if (s.empty()) return false; sscanf_s(s.c_str(),"%lu",&d); return true; }
	bool GetItem(float& d) { string s=GetItem(); if (s.empty()) return false; sscanf_s(s.c_str(),"%g",&d); return true; }
	bool GetItem(double& d) { string s=GetItem(); if (s.empty()) return false; sscanf_s(s.c_str(),"%lg",&d); return true; }
	bool GetItem(long long& d) { string s=GetItem(); if (s.empty()) return false; sscanf_s(s.c_str(),"%lli",&d); return true; }
	bool GetItem(unsigned long long& d) { string s=GetItem(); if (s.empty()) return false; sscanf_s(s.c_str(),"%llu",&d); return true; }
	bool GetItem(LPSTR sz, size_t n);
	bool GetItem(LPWSTR wsz, size_t n);
	bool GetItem(string& s) { s = GetItem(); return s.size()>0; }
	bool GetItem(wstring& ws) { string s = GetItem(); if (s.empty()) return false; ws = utf8wc(s); return true; }

	template<class T>
	bool GetItem(vector<T>& v)
	{
		size_t n; if (!GetItem(n)) return false;
		T d; v.clear(); v.reserve(n);
		for (size_t j=0; j<n; ++j)
			if (GetItem(d)) v.push_back(d);
			else return false;
		return true;
	}

	bool PutItem(bool f) { return PutItem(f ? 'T':'F'); }
	bool PutItem(char c);
	bool PutItem(wchar_t wc);
	bool PutItem(short d) { char sz[32]; sprintf_s(sz, "%hi", d); return PutItem(sz); }
	bool PutItem(int d) { char sz[32]; sprintf_s(sz, "%i", d); return PutItem(sz); }
	bool PutItem(long d) { char sz[32]; sprintf_s(sz, "%li", d); return PutItem(sz); }
	bool PutItem(unsigned short d) { char sz[32]; sprintf_s(sz, "%hu", d); return PutItem(sz); }
	bool PutItem(unsigned int d) { char sz[32]; sprintf_s(sz, "%u", d); return PutItem(sz); }
	bool PutItem(unsigned long d) { char sz[32]; sprintf_s(sz, "%lu", d); return PutItem(sz); }
	bool PutItem(float d) { char sz[32]; sprintf_s(sz, "%.9g", d); return PutItem(sz); }
	bool PutItem(double d) { char sz[32]; sprintf_s(sz, "%.17g", d); return PutItem(sz); }
	bool PutItem(long long d) { char sz[32]; sprintf_s(sz, "%lli", d); return PutItem(sz); }
	bool PutItem(unsigned long long d) { char sz[32]; sprintf_s(sz, "%llu", d); return PutItem(sz); }
	bool PutItem(LPCSTR s);
	bool PutItem(const string& s) { return PutItem(s.c_str()); }
	bool PutItem(LPCWSTR s) { return PutItem(utf8c(s).c_str()); }
	bool PutItem(const wstring& s) { return PutItem(utf8c(s).c_str()); }

	template<class T>
	bool PutItem(const vector<T>& v)
	{
		if (!PutItem(v.size())) return false;
		for (vector<T>::const_iterator i=v.begin(); i!=v.end(); ++i)
			if (!PutItem(*i)) return false;
		return true;
	}
	bool PutMarkerBegin(LPCSTR s);
	bool PutMarkerEnd(LPCSTR s);
	bool GetMarker(LPCSTR s);
	bool TryGetMarker(LPCSTR s);

	operator FILE*() const { return m_file; }

	bool GetLine(string& s);
	bool GetLine(wstring& s);

	static bool ReadText(vector<string>& text, LPCWSTR file);
	static bool ReadText(vector<string>& text, const wstring& file) { return ReadText(text,file.c_str()); }
	static bool ReadText(vector<wstring>& text, LPCWSTR file);
	static bool ReadText(vector<wstring>& text, const wstring& file) { return ReadText(text,file.c_str()); }

protected:
	inline bool Error(const char* s) { m_err = s; return false; }
	inline bool Error(const string& s) { m_err = s; return false; }
	inline bool Error(string s, errno_t e) { char sz[128]; strerror_s(sz, sizeof(sz), e); m_err = s + sz; return false; }
};

template<class T>
File& operator << (File& ff, const T& a) { ff.PutItem(a); return ff; }

template<class T>
File& operator >> (File& ff, T& a) { ff.GetItem(a); return ff; }

} } }
