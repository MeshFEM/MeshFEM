#include <MeshFEM/Types.hh>

// Print points as [x, y, z]
Eigen::IOFormat pointFormatter(Eigen::FullPrecision, Eigen::DontAlignCols,
        /* coeff separator */ "", /* row separator */ ", ",
        /* row prefix */ "", /* row suffix */ "", /* mat prefix */ "[",
        /* mat suffix */ "]");
