#ifndef LOAD_DENSE_MATRIX_HH
#define LOAD_DENSE_MATRIX_HH

#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <iterator>
#include <vector>

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> load_dense_matrix(const char *path) {
    size_t cols = 0, rows = 0;
    std::vector<T> entries;

    std::ifstream f(path);
    for (std::string line; std::getline(f, line);) {
        size_t offset = entries.size();
        std::stringstream stream(line);
        std::copy(std::istream_iterator<T>(stream),
                  std::istream_iterator<T>(),
                  std::back_inserter(entries));
        size_t currCols = entries.size() - offset;

        if (currCols == 0) continue;
        if (cols == 0) cols = currCols;
        if (currCols != cols)
            throw std::runtime_error("Nonuniform column size in the input file (ragged arrays are not supported)");

        rows++;
    }

	return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(entries.data(), rows, cols);
}

// Read a `rows x cols` matrix with entry type `T` from the input stream `is`.
// Doesn't do any shape checking: simply reads `rows * cols` entries and
// reshapes them into a matrix.
template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> load_matrix_from_stream(std::istream &is, size_t rows, size_t cols) {
    std::vector<T> entries;
    entries.reserve(rows * cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            T entry;
            is >> entry;
            entries.push_back(entry);
        }
    }
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(entries.data(), rows, cols);
}

#endif /* end of include guard: LOAD_DENSE_MATRIX_HH */
