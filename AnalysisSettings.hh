////////////////////////////////////////////////////////////////////////////////
// AnalysisSettings.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Stores (and saves/parses) all the settings for CSGFEM.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/01/2013 23:39:01
////////////////////////////////////////////////////////////////////////////////
#ifndef ANALYSIS_SETTINGS_HH
#define ANALYSIS_SETTINGS_HH
#include <boost/program_options.hpp>
#include <string>

namespace po = boost::program_options;

#include "GlobalTypes.hh"
#include "Quadrature.hh"

struct AnalysisSettings {
    AnalysisSettings()
        : solver("Gurobi"),
          Nx(40), Ny(40), borderWidth(1), quadrature(UNIFORM_QUADRATURE), quadraturePoints(81),
          cellOverlapThreshold(0.15), useMSBoundary(true), boundarySpacing(.02), kernelRadius(1.0),
          exactFullElements(true), antialiasedElements(false),
          massMatrixType(MASS_QUARTER_CELL),
          laplacianModes(false), consistentSigns(true), numModes(10),
          weakRegionsPerMode(5), weaknessCutoff(.95), abstrace(true), plusMinusObjective(true),
          totalForceBound(.1), pointwisePressureBound(.1),
          fixedTranslation(false), xTranslation(0.0), yTranslation(0.0),
          young_modulus(1.0), poisson_ratio(0.0), density(1.0) { }

    std::string solver;

    // Element settings
    size_t Nx, Ny;
    size_t borderWidth;
    QuadratureMethod quadrature;
    size_t quadraturePoints;
    double cellOverlapThreshold;
    bool   useMSBoundary;
    double boundarySpacing;
    double kernelRadius;

    bool exactFullElements; // Analytic integrals for elements when possible
    bool antialiasedElements;

    MassMatrixType massMatrixType;

    // True: use laplacian eigenfunctions instead of true stiffness eigenvectors
    bool laplacianModes, consistentSigns;
    size_t numModes;

    // Optimization Settings
    size_t weakRegionsPerMode;
    double weaknessCutoff;
    bool abstrace;
    bool plusMinusObjective;
    double totalForceBound;
    double pointwisePressureBound;

    // Translation Test
    bool fixedTranslation;
    double xTranslation, yTranslation; // unused only if fixedTranslation

    // Material Settings
    double young_modulus, poisson_ratio, density;

    // Memberwise comparator
    bool operator==(const AnalysisSettings &rhs) const {
        // Make sure new members haven't been added...
        // BOOST_STATIC_ASSERT((sizeof(AnalysisSettings) == 176)
        //         && "Settings members changed without updating comparator!");

        return ((solver == rhs.solver) && (Nx == rhs.Nx) && (Ny == rhs.Ny) &&
            (borderWidth == rhs.borderWidth) && (quadrature == rhs.quadrature) &&
            (quadraturePoints == rhs.quadraturePoints) &&
            (cellOverlapThreshold == rhs.cellOverlapThreshold) &&
            (useMSBoundary == rhs.useMSBoundary) &&
            (boundarySpacing == rhs.boundarySpacing) &&
            (kernelRadius == rhs.kernelRadius) &&
            (exactFullElements == rhs.exactFullElements) &&
            (antialiasedElements == rhs.antialiasedElements) &&
            (massMatrixType == rhs.massMatrixType) &&
            (laplacianModes == rhs.laplacianModes) &&
            (consistentSigns == rhs.consistentSigns) &&
            (numModes == rhs.numModes) &&
            (weakRegionsPerMode == rhs.weakRegionsPerMode) &&
            (weaknessCutoff == rhs.weaknessCutoff) &&
            (abstrace == rhs.abstrace) &&
            (plusMinusObjective == rhs.plusMinusObjective) &&
            (totalForceBound == rhs.totalForceBound) &&
            (pointwisePressureBound == rhs.pointwisePressureBound) &&
            (fixedTranslation == rhs.fixedTranslation) &&
            (xTranslation == rhs.xTranslation) &&
            (yTranslation == rhs.yTranslation) &&
            (young_modulus == rhs.young_modulus) &&
            (poisson_ratio == rhs.poisson_ratio) &&
            (density == rhs.density));
    }

    void getOptions(po::options_description &opts) const {
        opts.add_options()
            ("Nx", po::value<int>()->default_value(40), "Grid rows")
            ("Ny", po::value<int>()->default_value(40), "Grid columns")
            ("quadrature", po::value<std::string>()->default_value("uniform"), "Quadrature type")
            ("quadrature_points", po::value<int>()->default_value(81), "Number of quadrature points")
            ("cell_overlap_threshold", po::value<double>()->default_value(0.15), "Quad point fraction needed to qualify as a cell")
            ("ms_boundary", "Use marching squares boundary")
            ("boundary_spacing", po::value<double>()->default_value(.02), "Boundary point spacing (when use_ms_boundary is false)")
            ("mass_matrix_type", po::value<std::string>()->default_value("quarter_cell"), "Type of mass matrix")
            ("laplacian_modes", "Use laplacian eigenvectors as modes.")
            ("num_modes", po::value<int>()->default_value(10), "Number of modes to compute")
            ("weak_regions_per_mode", po::value<int>()->default_value(5), "Number of weak regions to extract per mode")
            ("weakness_cutoff", po::value<double>()->default_value(.95), "Stress norm percentile above which a cell is considered weak")
            ("total_force_bound", po::value<double>()->default_value(0.1), "F_tot: equality constraint for the total force")
            ("pointwise_pressure_bound", po::value<double>()->default_value(0.1), "p_max: maximum pressure at each boundary point")

            ("young_modulus", po::value<double>()->default_value(1.0), "Material's young modulus")
            ("poisson_ratio", po::value<double>()->default_value(0.0), "Material's poisson ratio")
            ("density", po::value<double>()->default_value(1.0), "Material's density")
        ;
    }
};

#endif // ANALYSIS_SETTINGS_HH
