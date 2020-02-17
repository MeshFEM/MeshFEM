#include <MeshFEM/SparseMatrices.hh>
#include <stdexcept>

using TMatrix = TripletMatrix<>;

int main(int argc, const char *argv[]) {
    if (argc != 3) {
        std::cerr << "usage: solve_system tmat.bin rhs.txt" << std::endl;
        exit(-1);
    }

    TMatrix mat;
    mat.readBinary(argv[1]);

    std::vector<Real> rhs;
    rhs.reserve(mat.m);
    {
        std::ifstream inFile(argv[2]);
        if (!inFile.is_open()) throw std::runtime_error(std::string("Couldn't open input file ") + argv[2]);
        double val;
        while (inFile >> val)
            rhs.push_back(val);
    }
    if (rhs.size() != mat.m) throw std::runtime_error("System size mismatch (" + std::to_string(rhs.size()) + " vs " + std::to_string(mat.m) + ")");

    // CholmodFactorizer fac(mat, true, false); // force supernodal, LL
    // auto result = fac.solve(rhs);

    SPSDSystem<Real> fac(mat);
    fac.setForceSupernodal(true);
    auto result = fac.solve(rhs);

    std::cout.precision(19);
    for (Real v : result)
        std::cout << v << "\n";

    return 0;
}
