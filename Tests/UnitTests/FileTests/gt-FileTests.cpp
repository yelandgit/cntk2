
#include "stdafx.h"
#include "FileTest.h"
#include "Common/Path.h"
//#include <Math/Matrix.h>
//#include <Math/CPUMatrix.h>
//#include <Math/Helpers.h>
#include "gtest/gtest.h"
#include "../GoogleTestHelper/TestDataHelper.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

class FileTests : public ::testing::Test, public TestDataHelper
{
public:
	FileTests() {}
	static void SetUpTestCase() { Prepare(__FILE__); }
	static void TearDownTestCase() {}

	void CreateTextFiles();
};

TEST_F(FileTests, WriteCloseRead)
{
	std::wstring file = utf8wc(TempPath("TestData1.txt"));

	File ff;
	ASSERT_TRUE(ff.Open(file, FILE_WRITE|FILE_TEXT)) << "Cannot create " << file;
	FileTest ft1; ft1.Init(); ff << ft1;
	ASSERT_TRUE(ff.Close()) << "Cannot close " << file;

	ASSERT_TRUE(ff.Open(file, FILE_READ|FILE_TEXT)) << "Cannot open " << file;
	FileTest ft2; ff >> ft2;
	ASSERT_TRUE(ft2.m_err.empty()) << "read error: " << ft2.m_err;
	ASSERT_TRUE(ff.Close()) << "Cannot close " << file;
	ASSERT_TRUE(ft1==ft2);
}

TEST_F(FileTests, WriteReadClose)
{
	std::wstring file = utf8wc(TempPath("TestData2.txt"));

	File ff;
	ASSERT_TRUE(ff.Open(file, FILE_WRITE|FILE_MODIFY|FILE_TEXT)) << "Cannot create " << file;
	FileTest ft1, ft2; ft1.Init();
	ff << ft1; ff.SetPosition(0); ff >> ft2;
	ASSERT_TRUE(ff.Close()) << "Cannot close " << file;
	ASSERT_TRUE(ft1==ft2);
}

struct ReferenceData
{
	LPCSTR		file;
	size_t		size;
	LPCSTR		data;
};

static ReferenceData refData[] = {
	// regular text
	{ "text-crlf.txt", 13, "aa\r\nbb\r\nccc\r\n" },
	{ "text-crlfx.txt", 11, "aa\r\nbb\r\nccc" },
	{ "text-lf.txt", 10, "aa\nbb\nccc\n" },
	{ "text-lfx.txt", 9, "aa\nbb\nccc" },
	// utf-8
	{ "text-utf8-crlf.txt", 16, "\xEF\xBB\xBF""aa\r\nbb\r\nccc\r\n" },
	{ "text-utf8-crlfx.txt", 14, "\xEF\xBB\xBF""aa\r\nbb\r\nccc" },
	{ "text-utf8-lf.txt", 13, "\xEF\xBB\xBF""aa\nbb\nccc\n" },
	{ "text-utf8-lfx.txt", 12, "\xEF\xBB\xBF""aa\nbb\nccc" },
	// utf-16
	{ "text-utf16-crlf.txt", 28, "\xFF\xFE""a\0a\0\r\0\n\0b\0b\0\r\0\n\0c\0c\0c\0\r\0\n\0" },
	{ "text-utf16-crlfx.txt", 24, "\xFF\xFE""a\0a\0\r\0\n\0b\0b\0\r\0\n\0c\0c\0c\0" },
	{ "text-utf16-lf.txt", 22, "\xFF\xFE""a\0a\0\n\0b\0b\0\n\0c\0c\0c\0\n\0" },
	{ "text-utf16-lfx.txt", 20, "\xFF\xFE""a\0a\0\n\0b\0b\0\n\0c\0c\0c\0" },
	{ "", 0, "" }
};

void FileTests::CreateTextFiles()
{
	FILE* ff;
	string root = TempPath("");
	for (int j=0; refData[j].size>0; ++j)
	{
		string file = root + refData[j].file;
		ASSERT_TRUE(fopen_s(&ff,file.c_str(),"wb")==0) << "Cannot create " << file;
		size_t n = fwrite(refData[j].data,1,refData[j].size,ff); fclose(ff);
		ASSERT_TRUE(n==refData[j].size) << "write error";
	}
}

template<class T>
static bool t_same(const vector<T>& v1, const vector<T>& v2)
{
	if (v1.size()!=v2.size()) return false;
	for (vector<T>::const_iterator i1=v1.begin(),i2=v2.begin(); i1!=v1.end(); ++i1,++i2)
		if (*i1!=*i2) return false;
	return true;
}

static void GetReferenceText(vector<string>& svec, vector<wstring>& wsvec)
{
	LPCSTR refer[] = { "aa", "bb", "ccc", "" };
	svec.clear(); wsvec.clear();
	for (int j=0; refer[j][0]; ++j)
	{
		svec.push_back(string(refer[j]));
		wsvec.push_back(utf8wc(refer[j]));
	}
}

TEST_F(FileTests, ReadText)
{
	vector<string> refsvec;
	vector<wstring> refwsvec;
	GetReferenceText(refsvec,refwsvec);

	CreateTextFiles();

	vector<string> stext;
	vector<wstring> wstext;
	string root = TempPath("");
	for (int j=0; refData[j].size>0; ++j)
	{
		string file = root + refData[j].file;
		cout << "  " << file << endl;

		ASSERT_TRUE(File::ReadText(stext,utf8wc(file).c_str())) << "File not found";
		ASSERT_EQ(stext.size(), refsvec.size()) << "strvec; Invalid size";
		ASSERT_TRUE(t_same(stext,refsvec)) << "strvec; Different text";

		ASSERT_TRUE(File::ReadText(wstext,utf8wc(file).c_str())) << "File not found";
		ASSERT_EQ(wstext.size(), refwsvec.size()) << "wstrvec; Invalid size";
		ASSERT_TRUE(t_same(wstext,refwsvec)) << "wstrvec; Different text";
	}
}

template<class T>
void ReadTextItems(LPCWSTR file, T& item, const T& ref)
{
	File ff;
	ASSERT_TRUE(ff.Open(file,FILE_READ|FILE_TEXT)) << "File not found; " << file;
	ASSERT_TRUE(ff.GetMarker("aa")) << "Marker not found; " << file;
	ASSERT_TRUE(ff.GetItem(item)) << "Item not found; " << file;
	ASSERT_TRUE(item==ref) << "Invalid item; " << file;
	ASSERT_TRUE(ff.GetMarker("ccc")) << "Marker not found; " << file;
}

TEST_F(FileTests, ReadItems)
{
	string sitem;
	wstring wsitem;
	string root = TempPath("");
	for (int j=0; refData[j].size>0; ++j)
	{
		wstring file = utf8wc(root + refData[j].file);
		ReadTextItems(file.c_str(), sitem, string("bb"));
		ReadTextItems(file.c_str(), wsitem, wstring(L"bb"));
	}
}

} } } }
