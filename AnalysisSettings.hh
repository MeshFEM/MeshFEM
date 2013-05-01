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
        : Nx(40), Ny(40), quadrature(UNIFORM_QUADRATURE), quadraturePoints(81),
          cellOverlapThreshold(0.15), useMSBoundary(false), boundarySpacing(.02),
          massMatrixType(MASS_QUARTER_CELL),
          laplacianModes(false), numModes(10), weakRegionsPerMode(5),
          weaknessCutoff(.95), totalForceBound(.1), pointwisePressureBound(.1),
          young_modulus(1.0), poisson_ratio(0.0), density(1.0) { }

    // Element settings
    size_t Nx, Ny;
    QuadratureMethod quadrature;
    size_t quadraturePoints;
    double cellOverlapThreshold;
    bool   useMSBoundary;
    double boundarySpacing;

    MassMatrixType massMatrixType;

    // True: use laplacian eigenfunctions instead of true stiffness eigenvectors
    bool laplacianModes;
    size_t numModes;

    // Optimization Settings
    size_t weakRegionsPerMode;
    double weaknessCutoff;
    double totalForceBound;
    double pointwisePressureBound;

    // Material Settings
    double young_modulus, poisson_ratio, density;

    void getOptions(po::options_description &opts) const {
        opts.add_options()
            ("Nx", po::value<int>()->default_value(40), "Grid rows")
            ("Ny", po::value<int>()->default_value(40), "Grid columns")
            ("quadrature", po::value<std::string>()->default_value("uniform"), "Quadrature type")
            ("quadrature_points", po::value<int>()->default_value(81), "Number of quadrature points")
            ("cell_overlap_threshold", po::value<double>()->default_value(0.15), "Quad point fraction needed to qualify as a cell")
            ("use_ms_boundary", po::value<bool>()->default_value(true), "Use marching squares boundary")
            ("boundary_spacing", po::value<double>()->default_value(.02), "Boundary point spacing (when use_ms_boundary is false)")
            ("mass_matrix_type", po::value<std::string>()->default_value("quarter_cell"), "Type of mass matrix")
            ("laplacian_modes", po::value<bool>()->default_value(true), "Use laplacian eigenvectors as modes.")
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
