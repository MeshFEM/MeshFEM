#include <MeshFEMSparse/BlockCSCHessian.hh>
#include <MeshFEMSparse/BlockCSCHessianDynCastWorkaround.hh> // NOLINT
#include <MeshFEMSparse/VarStructure.hh>

namespace MeshFEM {

// Defined out-of-line to ensure a single vtable is generated and exported
// by libMeshFEM (in an effort to resolve RTTI/dynamic_cast errors).
BlockCSCHessianBase::~BlockCSCHessianBase() = default;

void BlockCSCHessianBase::dumpBinaryToStream(std::ostream &os) const {
    if (!uniformBlockSize()) throw std::runtime_error("dumpBinaryToStream: non-uniform block size not supported");
    if (hasContiguousBlocks()) return cloneWithNoncontiguousBlocks()->dumpBinaryToStream(os); // For historical reasons, the output format is always the noncontiguous block format.

    size_t blockSize = minBlockSize();
    size_t numBlocks = vars().numBlocks();
    os.write(reinterpret_cast<const char *>(&blockSize), sizeof(blockSize));
    os.write(reinterpret_cast<const char *>(&numBlocks), sizeof(numBlocks));

    index_type snz = isSparsityOnly() ? 0 : scalarNNZ();
    if ((Ap.size() != size_t(n + 1)) || (size_t(Ai.size()) != size_t(nz)) || (Ax.size() != size_t(snz))) {
        std::cout << "Ap.size() = " << Ap.size() << std::endl;
        std::cout << "Ai.size() = " << Ai.size() << std::endl;
        std::cout << "Ax.size() = " << Ax.size() << std::endl;

        std::cout << "n = " << n << std::endl;
        std::cout << "nz = " << nz << std::endl;
        std::cout << "snz = " << snz << std::endl;
        throw std::runtime_error("Inconsistent matrix size metadata");
    }

    os.write((const char *) &  m, sizeof(index_type));
    os.write((const char *) &  n, sizeof(index_type));
    os.write((const char *) & nz, sizeof(index_type));
    os.write((const char *) &snz, sizeof(index_type)); // Differs from CSCMatrix format! The number of scalar entries stored in Ax.
    os.write((const char *) &symmetry_mode, sizeof(uint32_t));
    os.write((const char *) Ap.data(), Ap.size() * sizeof(index_type));
    os.write((const char *) Ai.data(), Ai.size() * sizeof(index_type));
    os.write((const char *) Ax.data(), Ax.size() * sizeof(double));
}

void BlockCSCHessianBase::readBinaryFromStream(std::istream &is) {
    index_type snz;
    is.read((char *) &  m, sizeof(index_type));
    is.read((char *) &  n, sizeof(index_type));
    is.read((char *) & nz, sizeof(index_type));
    is.read((char *) &snz, sizeof(index_type));
    is.read((char *) &symmetry_mode, sizeof(uint32_t));

    if ((symmetry_mode != SymmetryMode::NONE) &&
        (symmetry_mode != SymmetryMode::UPPER_TRIANGLE) &&
        (symmetry_mode != SymmetryMode::LOWER_TRIANGLE)) {
        throw std::runtime_error("Invalid symmetry_mode: " + std::to_string(uint32_t(symmetry_mode)));
    }

    Ap.resize(n + 1);
    Ai.resize(nz);
    Ax.resize(snz);

    is.read((char *) Ap.data(), Ap.size() * sizeof(snz));
    is.read((char *) Ai.data(), Ai.size() * sizeof(snz));
    is.read((char *) Ax.data(), Ax.size() * sizeof(snz));

    finalize();
    if (!isSparsityOnly() && (scalarNNZ() != size_t(snz)))  throw std::runtime_error("constructFromBinaryStream: scalarNNZ mismatch");
}

// Explicit template instantiations of BlockCSCHessian used by MeshFEM;
// instantiations for user code should be added in a separate source file in the
// user's project.
// Note: these apparently must be annotated with MESHFEM_EXPORT even though the
// template declaration was also annotated with MESHFEM_EXPORT....
template struct MESHFEM_EXPORT BlockCSCHessian<OptimizationVarStructure<1>, false>;
template struct MESHFEM_EXPORT BlockCSCHessian<OptimizationVarStructure<2>, false>;
template struct MESHFEM_EXPORT BlockCSCHessian<OptimizationVarStructure<3>, false>;
template struct MESHFEM_EXPORT BlockCSCHessian<OptimizationVarStructure<3, 1, 1>, false>;

template struct MESHFEM_EXPORT BlockCSCHessian<OptimizationVarStructure<1>, true>;
template struct MESHFEM_EXPORT BlockCSCHessian<OptimizationVarStructure<2>, true>;
template struct MESHFEM_EXPORT BlockCSCHessian<OptimizationVarStructure<3>, true>;
template struct MESHFEM_EXPORT BlockCSCHessian<OptimizationVarStructure<3, 1, 1>, true>;

// On GCC, this function must come after the explicit instantiations above
// because the implicit instantiations that are otherwise triggered inside have
// the wrong symbol visibility.
//      https://gcc.gnu.org/legacy-ml/gcc-help/2011-08/msg00109.html
//      warning: type attributes ignored after type is already defined [-Wattributes]
std::unique_ptr<BlockCSCHessianBase> BlockCSCHessianBase::constructFromBinaryStream(std::istream &is) {
    size_t blockSize, numBlocks;
    is.read(reinterpret_cast<char *>(&blockSize), sizeof(blockSize));
    is.read(reinterpret_cast<char *>(&numBlocks), sizeof(numBlocks));

    std::unique_ptr<BlockCSCHessianBase> result;
    if (blockSize == 1) result = BlockCSCHessian<OptimizationVarStructure<1>, false>::construct(OptimizationVarStructure<1>(numBlocks));
    if (blockSize == 2) result = BlockCSCHessian<OptimizationVarStructure<2>, false>::construct(OptimizationVarStructure<2>(numBlocks));
    if (blockSize == 3) result = BlockCSCHessian<OptimizationVarStructure<3>, false>::construct(OptimizationVarStructure<3>(numBlocks));
    if (!result) throw std::runtime_error("constructFromBinaryStream: uninstantiated block size");
    result->readBinaryFromStream(is);

    if (ContiguousBlocksDefault) return result->cloneWithContiguousBlocks();
    return result;
}

std::unique_ptr<BlockCSCHessianBase> BlockCSCHessianBase::fromScalar(const SuiteSparseMatrix &A) {
    auto result = BlockCSCHessian<OptimizationVarStructure<1>, false>::construct(OptimizationVarStructure<1>(size_t(A.m)));
    result->SuiteSparseMatrix::operator=(A);
    return result;
}

} // namespace MeshFEM
