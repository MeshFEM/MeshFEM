#ifndef ARGPARSE_HH
#define ARGPARSE_HH

#include <string>
#include <tuple>
#include <vector>
#include <MeshFEM/Types.hh>
#include <boost/optional.hpp>

// Filter invocation: (name, argument string)
using FilterInvocation = std::pair<std::string, std::string>;
// Parse command line arguments to get a .msh file path and a sequence of filters.
std::tuple<std::string, std::vector<FilterInvocation>, boost::optional<size_t>> parseCmdLine(int argc, char *argv[]);

// Parse a filter invocation's argument.
int parseIntArg(const std::string &arg);
double parseRealArg(const std::string &arg);

template<size_t N>
using VecHPCFix = Eigen::Matrix<Real, N, 1, 0>;

template<size_t N>
VecHPCFix<N> parseVectorArg(const std::string &arg);

template<size_t N>
std::vector<VecHPCFix<N>> parseVectorListArg(const std::string &arg);

#endif /* end of include guard: ARGPARSE_HH */
