#pragma once

////////////////////////////////////////////////////////////////////////////////
#include <vector>
#include <string>
////////////////////////////////////////////////////////////////////////////////

namespace MeshFEM {

	// Convert to lowercase
	std::string lowercase(std::string data);

	// Trim a string
	void trim(std::string &str);

	// Tests whether a string starts with a given prefix
	bool startswith(const std::string &str, const std::string &prefix);

	// Split a string into tokens
	std::vector<std::string> split(const std::string &str, const std::string &delimiters = " ");

	// Replace extension after the last "dot"
	std::string replace_ext(const std::string &filename, const std::string &newext);

} // namespace MeshFEM
