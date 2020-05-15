
#include "TestDataHelper.h"
#include "Common/StringUtils.h"
#include "Common/Directory.h"
#include "Common/Path.h"
#include "gtest/gtest.h"

string TestDataHelper::execPath;
string TestDataHelper::currPath;
string TestDataHelper::filePath;
string TestDataHelper::dataPath;
string TestDataHelper::tempPath;


void TestDataHelper::Prepare(string file, string data, string temp)
{
	execPath = Path(GetModuleName()).DriveDir();
	currPath = utf8c(GetCurrentDir()); if (*currPath.rbegin()!=DIRCHAR) currPath.push_back(DIRCHAR);
	filePath = Path(file).DriveDir();
	dataPath = filePath + data; if (*dataPath.rbegin()!=DIRCHAR) dataPath.push_back(DIRCHAR);
	tempPath = filePath + temp; if (*tempPath.rbegin()!=DIRCHAR) tempPath.push_back(DIRCHAR);
	if (temp.empty()) return;

	// prepare temp directory
	wstring path = utf8wc(tempPath);
	if (!RemoveAll(path,true,true)) GTEST_FATAL_FAILURE_("Remove ") << tempPath.c_str();
	if (!CreateDirs(path)) GTEST_FATAL_FAILURE_("Create ") << tempPath.c_str();
}

string TestDataHelper::DataPath(string path) { return IsFullPath(path) ? path : dataPath + path; }
string TestDataHelper::TempPath(string path) { return IsFullPath(path) ? path : tempPath + path; }
