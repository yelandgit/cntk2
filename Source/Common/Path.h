#pragma once

#include "StringUtils.h"
#include <string>
#include <iostream>

#ifdef _WIN32
#define DIRCHAR	'\\'
#else
#define DIRCHAR	'/'
#endif


struct Path
{
	std::string		drive;
	std::string		dir;
	std::string		name;
	std::string		ext;

	Path() {}
	Path(const char* path) { SetPath(path); }
	Path(const wchar_t* path) { SetPath(path); }
	Path(const std::string& path) { SetPath(path); }
	Path(const std::wstring& path) { SetPath(path); }

	Path(std::string path, std::string patt, bool force=false)  { SetPath(path); MergePath(patt, force); }
	Path(std::wstring path, std::wstring patt, bool force=false)  { SetPath(path); MergePath(patt, force); }

	Path& SetPath(const std::string& path) { SetPath(utf8wc(path)); return *this; }
	Path& SetPath(const std::wstring& path);

	std::string GetPath() const { return drive + dir + name + ext; }
	std::string DriveDir() const { return drive + dir; }
	std::string NameExt() const { return name + ext; }

	operator std::string() const { return drive + dir + name + ext; }

	Path& MergePath(const std::string& patt, bool force=false) { return MergePath(Path(patt),force); }
	Path& MergePath(const Path& patt, bool force=false);

	bool GoUp();
	void GoDown(const std::string& s) { dir += s; char c = *dir.rbegin(); if (c!='\\' && c!='/') dir.push_back(DIRCHAR); }

	bool operator == (const Path& p) { return drive==p.drive && dir==p.dir && name==p.name && ext==p.ext; }
};

inline std::ostream& operator << (std::ostream& os, const Path& path) { os << path.GetPath().c_str(); return os; }
