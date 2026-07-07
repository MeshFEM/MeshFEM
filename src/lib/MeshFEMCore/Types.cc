#include <MeshFEMCore/Types.hh>

namespace MeshFEM {

// Print points as [x, y, z]
Eigen::IOFormat pointFormatter(Eigen::FullPrecision, Eigen::DontAlignCols,
        /* coeff separator */ "", /* row separator */ ", ",
        /* row prefix */ "", /* row suffix */ "", /* mat prefix */ "[",
        /* mat suffix */ "]");

} // namespace MeshFEM
