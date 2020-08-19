#pragma once

////////////////////////////////////////////////////////////////////////////////
#include <vector>
#include <string>

#include <MeshFEM_export.h>
////////////////////////////////////////////////////////////////////////////////

namespace MeshFEM {

	// Convert to lowercase
	MESHFEM_EXPORT std::string lowercase(std::string data);

	// Trim a string
	MESHFEM_EXPORT void trim(std::string &str);

	// Tests whether a string starts with a given prefix
	MESHFEM_EXPORT bool startswith(const std::string &str, const std::string &prefix);

	// Split a string into tokens
	MESHFEM_EXPORT std::vector<std::string> split(const std::string &str, const std::string &delimiters = " ");

	// Replace extension after the last "dot"
	MESHFEM_EXPORT std::string replace_ext(const std::string &filename, const std::string &newext);

} // namespace MeshFEM
