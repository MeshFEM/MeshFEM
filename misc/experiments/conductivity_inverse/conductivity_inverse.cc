#include <MeshIO.hh>
#include <MSHFieldWriter.hh>

#include "Conductivity.hh"

using namespace std;

template<size_t N, size_t Deg>
void execute(const vector<MeshIO::IOVertex> &inVertices, 
             const vector<MeshIO::IOElement> &inElements) {

    FEMMesh<N, Deg, VectorND<N>> omega(inElements, inVertices);
    std::vector<Real> f(omega.numNodes()), a(omega.numNodes());

    for (auto n : omega.nodes()) {
        Real x = n->p[0], y = n->p[1];
        f[n.index()] = sin(2 * M_PI * x * y);
        a[n.index()] = 1.5 + cos(0.5 * M_PI * x * y);
    }

    MSHFieldWriter writer("out.msh", omega, false);

    // Compute forward solution and residual
    auto u = Conductivity::solveForwardProblem(omega, a, f);
    auto b = Conductivity::load(omega, f);
    auto L = Conductivity::forwardProblemMatrix(omega, a);
    auto r = ScalarField<Real>(L.apply(u)) - ScalarField<Real>(b); // Dirichlet load

    auto a_inferred = Conductivity::solveDirectInverseProblem(omega, u, a, f, r, writer);

    auto L_ainf = Conductivity::forwardProblemMatrix(omega, a_inferred);
    auto r_ainf = ScalarField<Real>(L_ainf.apply(u)) - ScalarField<Real>(b);
    auto L_1 = Conductivity::forwardProblemMatrix(omega, std::vector<Real>(omega.numNodes(), 1.0));
    auto r_1 = ScalarField<Real>(L_1.apply(u)) - ScalarField<Real>(b);

    // Compute inverse system residual
    auto M = Conductivity::directInverseProblemMatrix(omega, u);
    ScalarField<Real> ma = ScalarField<Real>(M.apply(a));
    ScalarField<Real> m_ainf = ScalarField<Real>(M.apply(a_inferred));

    ScalarField<Real> mr = ma - ScalarField<Real>(b);
    ScalarField<Real> mr_ainf = m_ainf - ScalarField<Real>(b);
    ScalarField<Real> mr_a1 = ScalarField<Real>(M.apply(std::vector<Real>(omega.numNodes(), 1.0))) - ScalarField<Real>(b);
    ScalarField<Real> bsurf = ScalarField<Real>(Conductivity::surfaceLoad(omega, u, a));

    writer.addField("f", ScalarField<Real>(f));
    writer.addField("a", ScalarField<Real>(a));
    writer.addField("u", ScalarField<Real>(u));
    writer.addField("a_inferred", ScalarField<Real>(a_inferred));
    writer.addField("a_infered residual", ScalarField<Real>(r_ainf));
    writer.addField("a residual", ScalarField<Real>(r));
    writer.addField("a=1 residual", ScalarField<Real>(r_1));

    writer.addField("M * a", ma);
    writer.addField("M * a_inf", m_ainf);
    writer.addField("Mr", mr);
    writer.addField("Mr_ainf", mr_ainf);
    // writer.addField("Mr a = 1", mr_a1);
    writer.addField("bsurf", bsurf);

    writer.addField("load", ScalarField<Real>(b));
}


int main(int argc, char *argv[])
{
    vector<MeshIO::IOVertex>  inVertices;
    vector<MeshIO::IOElement> inElements;

    // usage: mesh_path fem_degree
    string meshPath = argv[1];
    size_t deg = stoi(argv[2]);

    auto type = load(meshPath, inVertices, inElements, MeshIO::FMT_GUESS,
                     MeshIO::MESH_GUESS);

    // Infer dimension from mesh type.
    size_t dim;
    if      (type == MeshIO::MESH_TET) dim = 3;
    else if (type == MeshIO::MESH_TRI) dim = 2;
    else    throw std::runtime_error("Mesh must be pure triangle or tet.");

    auto exec = (dim == 3) ? ((deg == 2) ? execute<3, 2> : execute<3, 1>)
                           : ((deg == 2) ? execute<2, 2> : execute<2, 1>);
    exec(inVertices, inElements);
    return 0;
}
