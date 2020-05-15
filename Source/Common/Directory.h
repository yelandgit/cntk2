#pragma once

#include "Basics.h"


inline bool IsFullPath(const std::string& path) { return memcmp(path.c_str(),L"\\\\",2)==0 || (isalpha(path[0]) && path[1]==':');  }
inline bool IsFullPath(const std::wstring& path) { return memcmp(path.c_str(),L"\\\\",4)==0 || (iswalpha(path[0]) && path[1]==':');  }

bool ComparePath(const std::wstring& path, const std::wstring& patt, size_t i=0, size_t j=0);

bool PathUp(std::wstring& path);
bool GetFullPath(std::wstring& path);
bool CreateDirs(std::wstring path);

bool GetDirectory(wstrvec& dset, wstrvec& fset, std::wstring path);
bool RemoveAll(std::wstring path, bool subdir, bool view=false);

std::string CombinePath(const std::string& base, const std::string& path);
inline std::wstring CombinePath(const std::wstring& base, const std::wstring& path) { return utf8wc(CombinePath(utf8c(base),utf8c(path))); }

std::wstring GetCurrentDir();
inline std::wstring GetModuleName() { wchar_t path[_MAX_PATH]; ::GetModuleFileNameW(NULL, path, sizeof(path)/sizeof(path[0])); return std::wstring(path); }

inline bool PathExists(LPCWSTR path) { return ::GetFileAttributesW(path)!=INVALID_FILE_ATTRIBUTES; }
inline bool PathExists(const std::wstring& path) { return ::GetFileAttributesW(path.c_str())!=INVALID_FILE_ATTRIBUTES; }
inline bool DirExists(LPCWSTR path) { DWORD attr = ::GetFileAttributesW(path); return attr!=INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY)!=0; }
inline bool DirExists(const std::wstring& path) { DWORD attr = ::GetFileAttributesW(path.c_str()); return attr!=INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY)!=0; }

