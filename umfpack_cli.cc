////////////////////////////////////////////////////////////////////////////////
// suitesparse_cli.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//		Command-line interface to read in a matrix in triplet format and run
//		UMFPACK/Cholmod factorization for benchmarking (to compare against
//		openFTL's SuperLU solver).
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  03/28/2014 15:15:02
////////////////////////////////////////////////////////////////////////////////
#include "SparseMatrices.hh"
#include "Timer.hh"
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>

using namespace std;

////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on success)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    if (argc < 2 || argc > 3) {
        cout << "Usage: suitesparse_cli A.txt [UMFPACK|CHOLMOD]" << endl;
        return -1;
    }
    string APath(argv[1]);

    bool useCholmod = false;
    string solver("UMFPACK");
    if (argc == 3) {
        solver = argv[2];
        transform(solver.begin(), solver.end(), solver.begin(), ::toupper);
        if (solver == "CHOLMOD")
            useCholmod = true;
        else if (solver != "UMFPACK") {
            cout << "Warning: solver must be UMFPACK or CHOLMOD. Using UMFPACK."
                 << endl;
            solver = "UMFPACK";
        }
    }

    TripletMatrix<Triplet<double> > A;
    ifstream inFile(APath);
    if (!inFile.is_open()) {
        cout << "Failed to open input file '" << APath << '\'' << endl;
        return -1;
    }
    
    A.read(inFile);
    cout << "Factoring " << A.m << "x" << A.n << " matrix with " << A.nnz()
         << " nonzeros" << endl;

    Timer timer;

    timer.startSection("Full " + solver + " solve");
    timer.start("to SuiteSparseMatrix");
    SuiteSparseMatrix ssA(A);
    timer.stop("to SuiteSparseMatrix");

    if (useCholmod) {
        timer.start("Factorize");
        CholmodFactorizer factors(ssA);
        timer.stop("Factorize");

        timer.start("Solve");
        vector<double> b(A.m, 1.0), x;
        factors.solve(b, x);
        timer.stop("Solve");
    }
    else {
        timer.start("Factorize");
        UmfpackFactorizer factors(ssA);
        timer.stop("Factorize");

        timer.start("Solve");
        vector<double> b(A.m, 1.0), x;
        factors.solve(b, x);
        timer.stop("Solve");
    }

    timer.stopSection("Full " + solver + " solve");

    timer.report(cout);

    return 0;
}
