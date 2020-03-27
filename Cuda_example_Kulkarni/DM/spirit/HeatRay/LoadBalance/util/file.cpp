#include <util/file.h>

#include <fstream>

using namespace std;

namespace util
{

bool fileExists(const util::string& filename)
{
	ifstream fin(filename.c_str(), ios::in | ios::binary);
	bool result = fin;
	fin.close();
	return result;
}

}
