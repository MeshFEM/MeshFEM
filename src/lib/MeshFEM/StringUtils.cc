////////////////////////////////////////////////////////////////////////////////
#include "StringUtils.hh"
#include <algorithm>
#include <cctype>
#include <locale>
////////////////////////////////////////////////////////////////////////////////

namespace {

    // Trim function from:
    // https://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring

    // trim from start (in place)
    void ltrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
            return !std::isspace(ch);
        }));
    }

    // trim from end (in place)
    void rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
            return !std::isspace(ch);
        }).base(), s.end());
    }

}

////////////////////////////////////////////////////////////////////////////////

namespace MeshFEM {

// Convert to lowercase
std::string lowercase(std::string data) {
    std::transform(data.begin(), data.end(), data.begin(), ::tolower);
    return data;
}

// trim from both ends (in place)
void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

// Tests whether a string starts with a given prefix
bool startswith(const std::string &str, const std::string &prefix) {
    return (str.compare(0, prefix.size(), prefix) == 0);
}

// Split a string into tokens
std::vector<std::string> split(const std::string &str, const std::string &delimiters) {
    // Skip delimiters at beginning.
    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    // Find first "non-delimiter".
    std::string::size_type pos     = str.find_first_of(delimiters, lastPos);

    std::vector<std::string> tokens;
    while (std::string::npos != pos || std::string::npos != lastPos) {
        // Found a token, add it to the vector.
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        // Skip delimiters.  Note the "not_of"
        lastPos = str.find_first_not_of(delimiters, pos);
        // Find next "non-delimiter"
        pos = str.find_first_of(delimiters, lastPos);
    }

    return tokens;
}

// Replace extension after the last "dot"
std::string replace_ext(const std::string &filename, const std::string &newext) {
    std::string ext = "";
    if (!newext.empty()) {
        ext = (newext[0] == '.' ? newext : "." + newext);
    }
    size_t lastdot = filename.find_last_of(".");
    if (lastdot == std::string::npos) {
        return filename + ext;
    }
    return filename.substr(0, lastdot) + ext;
}

} // namespace MeshFEM

