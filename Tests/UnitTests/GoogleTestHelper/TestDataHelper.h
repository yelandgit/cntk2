#pragma once

#include <string>
using namespace std;

class TestDataHelper
{
public:
	static string	execPath;
	static string	currPath;
	static string	filePath;
	static string	dataPath;
	static string	tempPath;

	static void Prepare(string file, string data="TestData", string temp="TestTemp");
	static string DataPath(string path);
	static string TempPath(string path);
};
