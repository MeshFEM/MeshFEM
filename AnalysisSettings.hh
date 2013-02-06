////////////////////////////////////////////////////////////////////////////////
// AnalysisSettings.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Stores all the settings for modal/weakness analysis.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/01/2013 23:39:01
////////////////////////////////////////////////////////////////////////////////
#ifndef ANALYSIS_SETTINGS_HH
#define ANALYSIS_SETTINGS_HH

#include "GlobalTypes.hh"

struct AnalysisSettings {
    AnalysisSettings()
        : Nx(10), Ny(10), quadrature(UNIFORM_QUADRATURE), quadraturePoints(9),
          lumpedMass(false), laplacianModes(false), numModes(10),
          weakRegionsPerMode(5), weaknessCutoff(.95),
          young_modulus(1.0), poisson_ratio(0.0), density(1.0) { }

    // Element settings
    int Nx, Ny;
    QuadratureMethod quadrature;
    int quadraturePoints;
    bool lumpedMass, gaussNodes;

    // True: use laplacian eigenfunctions instead of true stiffness eigenvectors
    bool laplacianModes;
    int numModes;

    // Optimization Settings
    int weakRegionsPerMode;
    double weaknessCutoff;

    // Material Settings
    double young_modulus, poisson_ratio, density;
};

#endif // ANALYSIS_SETTINGS_HH
