#pragma once

////////////////////////////////////////////////////////////////////////////////
#include <vector>
#include <string>
////////////////////////////////////////////////////////////////////////////////

namespace MeshFEM {

	// Trim a string
	void trim(std::string &str);

	// Split a string into tokens
	std::vector<std::string> split(const std::string &str, const std::string &delimiters = " ");

} // namespace MeshFEM
