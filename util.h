////////////////////////////////////////////////////////////////////////////////
// util.h
////////////////////////////////////////////////////////////////////////////////
/*! @file
//        Miscellaneous utilities
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  05/10/2012 19:41:45
////////////////////////////////////////////////////////////////////////////////
#ifndef UTIL_H
#define UTIL_H

#include <sys/stat.h>
#include <string>

////////////////////////////////////////////////////////////////////////////
/*! Check if a file exists
//  @param[in]  filename   description
//  @return     true if the file exists
*///////////////////////////////////////////////////////////////////////////
bool fileExists(const std::string &filename)
{
    struct stat buf;
    if (stat(filename.c_str(), &buf) != -1) {
        return true;
    }
    return false;
}

////////////////////////////////////////////////////////////////////////////
/*! Get the next available (non-existant) file in a numbered sequence:
//  base%iext
//  @param[in]  base    filename base
//  @param[in]  ext     filename extension
//  @return     available filename
*///////////////////////////////////////////////////////////////////////////
std::string nextNewFile(const std::string &base, const std::string &ext)
{
    string result;
    int i = 0;
    do {
        stringstream ss;
        ss << base << i << ext;
        result = ss.str();
        ++i;
    } while (fileExists(result));
    return result;
}

#endif // UTIL_H
