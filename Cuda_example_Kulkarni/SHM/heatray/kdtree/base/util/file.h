#pragma once

#include <util/string.h>

/**
 * Utility functions and classes that are not specific to graphics or math.
 */
namespace util
{

/**
 * Check whether a file exists.
 */
bool fileExists(const util::string& filename);

}
