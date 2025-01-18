#ifndef TINYADPARAMETRIZATION_HH
#define TINYADPARAMETRIZATION_HH

#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/Parametrization.hh>

#include <TinyAD/ScalarFunction.hh>
#include <TinyAD/Utils/NewtonDirection.hh>
#include <TinyAD/Utils/NewtonDecrement.hh>
#include <TinyAD/Utils/LineSearch.hh>

namespace TinyADParametrization{

using Mesh = FEMMesh<2, 1, Vector3D>; // Piecewise linear triangle mesh embedded in R^3
using UVMap = Eigen::Matrix<Real, Eigen::Dynamic, 2, Eigen::ColMajor>;
using NDMap = Eigen::MatrixXd;
using V2d   = Eigen::Vector2d;
using V3d   = Eigen::Vector3d;
using VXd   = Eigen::VectorXd;
using M2d   = Eigen::Matrix<Real, 2, 2>;


MESHFEM_EXPORT
NDMap symmdsParamTinyAD(const Mesh &mesh, NDMap &uv_init) {
    const size_t nn = mesh.numNodes();
    const size_t num_ele = mesh.numElements();
    if (size_t(uv_init.rows()) != nn) throw std::runtime_error("Invalid uv initialization size");

    size_t numCompoents = uv_init.cols();
    NDMap result(nn, numCompoents);

    // pre-compute triangle rest shapes in local coordinate systems
    std::vector<M2d> rest_shapes(num_ele);
    for (const auto e : mesh.elements()) {
        // Get 3D vertex positions
        V3d ar_3d = e.node(0)->p;
        V3d br_3d = e.node(1)->p;
        V3d cr_3d = e.node(2)->p;

        // Set up local 2D coordinate system
        V3d n = (br_3d - ar_3d).cross(cr_3d - ar_3d);
        V3d b1 = (br_3d - ar_3d).normalized();
        V3d b2 = n.cross(b1).normalized();

        // Express a,b,c in local 2D coordinate system
        V2d ar_2d(0.0, 0.0);
        V2d br_2d((br_3d - ar_3d).dot(b1), 0.0);
        V2d cr_2d((cr_3d - ar_3d).dot(b1), (cr_3d - ar_3d).dot(b2));

        // save 2-by-2 matrix with edge vectors as columns
        rest_shapes[e.index()] = TinyAD::col_mat(br_2d - ar_2d, cr_2d - ar_2d);
    }

    // set up function with 2D vertex positions as variables.
    auto func = TinyAD::scalar_function<2>(TinyAD::range(nn));

    // Add objective term per face. Each connecting 3 vertiecs.
    func.add_elements<3>(TinyAD::range(num_ele), [&] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        // Evaluate element using either double or TinyAD::Double
        using T = TINYAD_SCALAR_TYPE(element);

        // Get variable 2D vertex positions
        Eigen::Index f_idx = element.handle;
        Eigen::Vector2<T> a = element.variables(mesh.element(f_idx).vertex(0).index());
        Eigen::Vector2<T> b = element.variables(mesh.element(f_idx).vertex(1).index());
        Eigen::Vector2<T> c = element.variables(mesh.element(f_idx).vertex(2).index());

        // Triangle flipped?
        Eigen::Matrix2<T> M = TinyAD::col_mat(b - a, c - a);
        if (M.determinant() <= 0.0)
            return (T)INFINITY;
        
        // Get constant 2D rest shape of f
        M2d Mr = rest_shapes[f_idx];
        double A = 0.5 * Mr.determinant();

        // Computer symmetric Dirichlet energy
        Eigen::Matrix2<T> J = M * Mr.inverse();
        return 0.5 * A * (J.squaredNorm() + J.inverse().squaredNorm());
    });

    // Assemble inital x vector from P matrix.
    // x_from_data(...) takes a lambda function that maps
    // each variable handle (vertex index) to its initial 2D value (Eigen::Vector2d).
    VXd x = func.x_from_data([&] (int v_idx) {
        return uv_init.row(v_idx);
    });

    // Projected Newton
    TinyAD::LinearSolver solver;
    int max_iters = 1000;
    double convergence_eps = 1e-2;
    for (int i = 0; i < max_iters; ++i)
    {
        auto [f, g, H_proj] = func.eval_with_hessian_proj(x);
        double g_norm = g.norm();
        TINYAD_DEBUG_OUT("Energy in iteration " << i << ": " << f);
        TINYAD_DEBUG_OUT("Gradient Norm in iteration " << i << ": " << g_norm);
        VXd d = TinyAD::newton_direction(g, H_proj, solver);
        if (TinyAD::newton_decrement(d, g) < convergence_eps)
            break;
        x = TinyAD::line_search(x, d, f, g, func);
    }
    //TINYAD_DEBUG_OUT("Final energy: " << func.eval(x));
    auto final_obj_grad = func.eval_with_gradient(x);
    double final_obj = std::get<0>(final_obj_grad);
    double final_grad_norm = (std::get<1>(final_obj_grad)).norm();
    TINYAD_DEBUG_OUT("Final energy: " << final_obj);
    TINYAD_DEBUG_OUT("Final gradient norm: " << final_grad_norm);

    // Write final x vector to P matrix.
    // x_to_data(...) takes a lambda function that writes the final value
    // of each variable (Eigen::Vector2d) back to our P matrix.
    func.x_to_data(x, [&] (int v_idx, const V2d& p) {
        result.row(v_idx) = p;
    });

    return result;
}


}



#endif /* end of include guard: TINYADPARAMETRIZATION_HH */