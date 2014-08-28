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
#include <sstream>

////////////////////////////////////////////////////////////////////////////
/*! Get the file extension for a path.
//  @param[in]  path
//  @return     file extension including initial period.
*///////////////////////////////////////////////////////////////////////////
inline std::string fileExtension(const std::string &path) {
    size_t last = path.find_last_of('.');
    return (last == std::string::npos) ? "" : path.substr(last);
}

////////////////////////////////////////////////////////////////////////////
/*! Check if a file exists
//  @param[in]  filename   description
//  @return     true if the file exists
*///////////////////////////////////////////////////////////////////////////
inline bool fileExists(const std::string &filename)
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
inline std::string nextNewFile(const std::string &base, const std::string &ext)
{
    std::string result;
    int i = 0;
    do {
        std::stringstream ss;
        ss << base << i << ext;
        result = ss.str();
        ++i;
    } while (fileExists(result));
    return result;
}

#endif // UTIL_H
