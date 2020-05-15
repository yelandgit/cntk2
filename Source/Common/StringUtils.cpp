
#include "stdafx.h"
#include "StringUtils.h"
#include <windows.h>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

using namespace std;


string unistr(const wchar_t* ws, int cp)
{
	int nch = ws[0] ? WideCharToMultiByte(cp, 0, ws, -1, NULL, 0, NULL, NULL) : 0;
	if (nch==0) return "";

	CHAR buff[256];
	if (nch <= sizeof(buff)/sizeof(CHAR))
	{
		WideCharToMultiByte(cp, 0, ws, -1, buff, nch, NULL, NULL);
		return buff;
	}
	LPSTR temp = new CHAR[nch];
	WideCharToMultiByte(cp, 0, ws, -1, temp, nch, NULL, NULL);
	string s(temp); delete[] temp;
	return s;
}

wstring uniwstr(const char* s, int cp)
{
	int nch = s[0] ? MultiByteToWideChar(cp, 0, s, -1, NULL, 0) : 0;
	if (nch==0) return L"";

	WCHAR buff[256];
	if (nch <= sizeof(buff)/sizeof(WCHAR))
	{
		MultiByteToWideChar(cp, 0, s, -1, buff, nch);
		return buff;
	}
	LPWSTR temp = new WCHAR[nch];
	MultiByteToWideChar(cp, 0, s, -1, temp, nch);
	wstring w(temp); delete[] temp;
	return w;
}

size_t tokens(strvec& v, const string& s, const char* sep)
{
	string item; v.clear();
	for (string::const_iterator i=s.begin(); i!=s.end(); ++i)
	{
		if (strchr(sep,*i)==NULL) { item.push_back(*i); continue; }
		if (item.size()>0) { v.push_back(item); item.clear(); }
	}
	if (item.size()>0) v.push_back(item);
	return v.size();
}

void xview(ostream& os, const void* lp, size_t n, int m)
{
	LPBYTE p = LPBYTE(lp);
	for (size_t j=0; j<n; j+=16,p+=16)
	{
		size_t ncb = n - j;
		char sz[32]; sprintf_s(sz,32,"%08x:",int(j)); os << sz;
		for (size_t i=0; i<16; ++i)
		{
			if ((i%m)==0) os << ' ';
			if (i<ncb) { sprintf_s(sz,32,"%02x",p[i]); os << sz; }
			else os << "..";
		}
		os << " |";
		for (size_t i=0; i<16; ++i)
		{
			if (i>=ncb) os << ' ';
			else if (isprint(p[i])) os << p[i];
			else os << '.';
		}
		os << "|" << endl;
	}
}
