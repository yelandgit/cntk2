
#include "stdafx.h"
#include "Path.h"
#include <windows.h>
using namespace std;


Path& Path::SetPath(const wstring& path)
{
	wchar_t pdrive[_MAX_DRIVE];
	wchar_t pdir[_MAX_DIR];
	wchar_t pname[_MAX_FNAME];
	wchar_t pext[_MAX_EXT];
	_wsplitpath_s(path.c_str(), pdrive, sizeof(pdrive)/sizeof(wchar_t), pdir, sizeof(pdir)/sizeof(wchar_t), 
							pname, sizeof(pname)/sizeof(wchar_t), pext, sizeof(pext)/sizeof(wchar_t));
	drive = utf8c(pdrive);
	dir = utf8c(pdir);
	name = utf8c(pname);
	ext = utf8c(pext);
	return *this;
}

Path& Path::MergePath(const Path& patt, bool force)
{
	if (force)
	{
		if (patt.drive.size()>0) drive = patt.drive;
		if (patt.dir.size()>0) dir = patt.dir;
		if (patt.name.size()>0) name = patt.name;
		if (patt.ext.size()>0) ext = patt.ext;
	}
	else
	{
		if (drive.size()==0) drive = patt.drive;
		if (dir.size()==0) dir = patt.dir;
		if (name.size()==0) name = patt.name;
		if (ext.size()==0) ext = patt.ext;
	}
	return *this;
}

bool Path::GoUp()
{
	if (dir.empty()) return false;
	size_t k = dir.find_last_of("\\/", dir.size()-2);
	if (k==string::npos) dir.clear();
	else if (k==0 && dir.size()==1) return false;
	else dir.erase(k+1);
	return true;
}
