////////////////////////////////////////////////////////////////////////////////
// MatrixRecorder.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  A global singleton used to record matrices and sparsity patterns for
//  testing and debugging.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  08/15/2025 16:43:26
*///////////////////////////////////////////////////////////////////////////////
#ifndef MATRIXRECORDER_HH
#define MATRIXRECORDER_HH

#include <MeshFEM_export.h>
#include <MeshFEMSparse/BlockCSCHessian.hh>
#include <string>
#include <stdexcept>

namespace MeshFEM {

struct MatrixRecorder {
    // If `path` is nonempty, record the matrices passed to symbolic and numeric factorization routines.
    void recordMatrices(const std::string &directory_path, bool symbolic = true, bool numeric = true, bool dynamic = false) {
        if (directory_path.empty()) throw std::runtime_error("MatrixRecorder: Must pass a nonempty path");
        if (symbolic + numeric == 0) throw std::runtime_error("MatrixRecorder: Must record at least one type of matrix (symbolic or numeric)");
        m_matrix_dump_path = directory_path;
        m_recordingNumeric = numeric;
        m_recordingSymbolic = symbolic;
        m_recordingDynamicSparsity = dynamic;
        resetIDs();
    }

    void stopRecordingMatrices() { m_matrix_dump_path.clear(); }
    bool recordingMatrices() const { return !m_matrix_dump_path.empty(); }
    static std::string symbolicMatrixFileName(size_t i) { return "symbolic_mat_" + m_matrixIdString(i) + ".bin"; }
    static std::string  numericMatrixFileName(size_t i) { return "numeric_mat_"  + m_matrixIdString(i) + ".bin"; }
    static std::string     pinnedVarsFileName(size_t i) { return "pinned_vars_"  + m_matrixIdString(i) + ".txt"; }

    static std::string symbolicMatrixFileName(size_t i, const std::string &suffix) { return "symbolic_mat_" + m_matrixIdString(i) + suffix + ".bin"; }

    void resetIDs() {
        m_matrixId = 0;
        m_numSparsityUpdateCalls = 0;
        m_dynamicSparsityCounter = 0;
    }

    void recordNumeric(const BlockCSCHessianBase &mat) {
        if (!recordingMatrices() || !m_recordingNumeric) return;
        mat.dumpBinaryToFile(m_matrix_dump_path + "/" + numericMatrixFileName(m_generateMatrixId()));
    }

    void recordStaticSparsity(const BlockCSCHessianBase &mat) {
        if (!recordingMatrices() || !m_recordingSymbolic) return;
        mat.dumpBinaryToFile(m_matrix_dump_path + "/static_sparsity_pattern.bin");
    }

    void recordDynamicSparsity(const BlockCSCHessianBase *mat) {
        if (!recordingMatrices() || !m_recordingSymbolic || !m_recordingDynamicSparsity) return;
        if (mat == nullptr) { ++m_dynamicSparsityCounter; return; } // Skip (but count) empty matrices.
        mat->dumpBinaryToFile(m_matrix_dump_path + "/dynamic_sparsity_pattern_" + m_matrixIdString(m_dynamicSparsityCounter++) + ".bin");
    }

    void recordSymbolic(const BlockCSCHessianBase &mat, const std::vector<size_t> &pinnedVars) {
        if (!recordingMatrices() || !m_recordingSymbolic) return;

        size_t id = m_generateMatrixId();
        mat.dumpBinaryToFile(m_matrix_dump_path + "/" + symbolicMatrixFileName(id, "_from_update_" + std::to_string(int(m_numSparsityUpdateCalls) - 1)));

        std::ofstream varsFile(m_matrix_dump_path + "/" + pinnedVarsFileName(id));
        for (size_t v : pinnedVars) varsFile << v << std::endl;
    }

    // Record the number of calls to updateSparsityPattern; this is needed
    // for reconstructing the correct cached entry ages (i.e., the number of times
    // SparsityLRU::increaseAgeOfOldEntries has been called between pattern updates)
    // when analyzing recorded matrices.
    void countSparsityUpdateCall() { ++m_numSparsityUpdateCalls; }

    bool recordingDynamicSparsity() const { return m_recordingDynamicSparsity; }

protected:
    // An increasing identifier used to sequence each matrix written
    // to disk during recording (symbolic and numeric).
    // We use the same ordering for both types since to that we easily
    // know which numeric matrices correspond to which sparsity patterns.
    static std::string m_matrixIdString(size_t id) {
        size_t n_zero = 4;
        std::string padded_num = std::to_string(id);
        padded_num = std::string(n_zero - std::min(n_zero, padded_num.length()), '0') + padded_num;
        return padded_num;
    }

    size_t m_generateMatrixId() { if (m_recordOnlyMostRecentMatrix) return m_matrixId; else return m_matrixId++; }

    std::string m_matrix_dump_path;

    bool m_recordingNumeric = false, m_recordingSymbolic = false, m_recordingDynamicSparsity;
    size_t m_matrixId = 0;
    size_t m_dynamicSparsityCounter = 0;
    size_t m_numSparsityUpdateCalls = 0; // Used to track the number of times the sparsity pattern has been updated.

    // If true, when recording the matrices, we overwrite the existing recorded matrices rather than writing fresh ones.
    // This is implemented by leaving `m_matrixId` at `0`.
    bool m_recordOnlyMostRecentMatrix = false;
};

MESHFEM_EXPORT extern MatrixRecorder g_matrixRecorder;

} // namespace MeshFEM

#endif /* end of include guard: MATRIXRECORDER_HH */
