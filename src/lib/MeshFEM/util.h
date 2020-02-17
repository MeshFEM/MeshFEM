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
#include <iostream>
#include <algorithm>
#include <functional>
#include <cctype>
#include <locale>

////////////////////////////////////////////////////////////////////////////////
/*! Reads a data line from ASCII file (skipping whitespace and comment lines).
//  @param[in]  is   input stream to read from
//  @param[out] line string output to hold data line
//  @return     reference to input stream for operator chaining
*///////////////////////////////////////////////////////////////////////////////
inline std::istream &getDataLine(std::istream &is, std::string &line) {
    do  {
        std::getline(is >> std::ws, line);

        // Allow windows line endings:
        if (!line.empty() && *line.rbegin() == '\r') {
            line.erase(line.length() - 1, 1);
        }
    } while (is && (line[0] == '#'));
    return is;
}

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

// http://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring
// trim from start
inline std::string ltrim(std::string s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int c) { return !std::isspace(c); }));
    return s;
}

// trim from end
inline std::string rtrim(std::string s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int c) { return !std::isspace(c); }).base(), s.end());
    return s;
}

// trim from both ends
inline std::string trim(std::string s) {
    return ltrim(rtrim(s));
}

#endif // UTIL_H
