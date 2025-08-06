////////////////////////////////////////////////////////////////////////////////
// FlipAvoidingStepLength.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Adapted from Michael Rabinovich's implementation here:
//      https://github.com/libigl/libigl/blob/main/include/igl/flip_avoiding_line_search.cpp
//  with some modification to Matrix types for efficiency and parallelization.
*///////////////////////////////////////////////////////////////////////////////
#ifndef FLIPAVOIDINGSTEPLENGTH_HH
#define FLIPAVOIDINGSTEPLENGTH_HH

#include <MeshFEM/newton_optimizer/FeasibleStepLengthComputer.hh>
#include <MeshFEM/GlobalBenchmark.hh>
#include <MeshFEM/Parallelism.hh>
#include <tbb/parallel_reduce.h>

namespace flip_avoiding {
    //---------------------------------------------------------------------------
    // x - array of size 3
    // In case 3 real roots: => x[0], x[1], x[2], return 3
    //         2 real roots: x[0], x[1],          return 2
    //         1 real root : x[0], x[1] ± i*x[2], return 1
    // http://math.ivanovo.ac.ru/dalgebra/Khashin/poly/index.html
    inline int SolveP3(std::vector<double>& x,double a,double b,double c)
    { // solve cubic equation x^3 + a*x^2 + b*x + c
      double a2 = a*a;
        double q  = (a2 - 3*b)/9;
      double r  = (a*(2*a2-9*b) + 27*c)/54;
        double r2 = r*r;
      double q3 = q*q*q;
      double A,B;
        if(r2<q3)
        {
          double t=r/sqrt(q3);
          if( t<-1) t=-1;
          if( t> 1) t= 1;
          t=acos(t);
          a/=3; q=-2*sqrt(q);
          x[0]=q*cos(t/3)-a;
          x[1]=q*cos((t+(2*M_PI))/3)-a;
          x[2]=q*cos((t-(2*M_PI))/3)-a;
          return(3);
        }
        else
        {
          A =-pow(fabs(r)+sqrt(r2-q3),1./3);
          if( r<0 ) A=-A;
          B = A==0? 0 : q/A;

          a/=3;
          x[0] =(A+B)-a;
          x[1] =-0.5*(A+B)-a;
          x[2] = 0.5*sqrt(3.)*(A-B);
          if(fabs(x[2])<1e-14)
          {
            x[2]=x[1]; return(2);
          }
          return(1);
        }
    }

    inline double get_smallest_pos_quad_zero(double a,double b, double c)
    {
      double t1, t2;
      if(std::abs(a) > 1.0e-10)
      {
        double delta_in = pow(b, 2) - 4 * a * c;
        if(delta_in <= 0)
        {
          return INFINITY;
        }

        double delta = sqrt(delta_in); // delta >= 0
        if(b >= 0) // avoid subtracting two similar numbers
        {
          double bd = - b - delta;
          t1 = 2 * c / bd;
          t2 = bd / (2 * a);
        }
        else
        {
          double bd = - b + delta;
          t1 = bd / (2 * a);
          t2 = (2 * c) / bd;
        }

        assert (std::isfinite(t1));
        assert (std::isfinite(t2));

        if(a < 0) std::swap(t1, t2); // make t1 > t2
        // return the smaller positive root if it exists, otherwise return infinity
        if(t1 > 0)
        {
          return t2 > 0 ? t2 : t1;
        }
        else
        {
          return INFINITY;
        }
      }
      else
      {
        if(b == 0) return INFINITY; // just to avoid divide-by-zero
        t1 = -c / b;
        return t1 > 0 ? t1 : INFINITY;
      }
    }

    template<class Derived1, class Derived2>
    inline double get_min_pos_root_2D(const Eigen::MatrixBase<Derived1> &uv,
                                      const Eigen::MatrixXi &F,
                                      const Eigen::MatrixBase<Derived2> &d, int f)
    {
    /*
          Finding the smallest timestep t s.t a triangle get degenerated (<=> det = 0)
          The following code can be derived by a symbolic expression in matlab:

          Symbolic matlab:
          U11 = sym('U11');
          U12 = sym('U12');
          U21 = sym('U21');
          U22 = sym('U22');
          U31 = sym('U31');
          U32 = sym('U32');

          V11 = sym('V11');
          V12 = sym('V12');
          V21 = sym('V21');
          V22 = sym('V22');
          V31 = sym('V31');
          V32 = sym('V32');

          t = sym('t');

          U1 = [U11,U12];
          U2 = [U21,U22];
          U3 = [U31,U32];

          V1 = [V11,V12];
          V2 = [V21,V22];
          V3 = [V31,V32];

          A = [(U2+V2*t) - (U1+ V1*t)];
          B = [(U3+V3*t) - (U1+ V1*t)];
          C = [A;B];

          solve(det(C), t);
          cf = coeffs(det(C),t); % Now cf(1),cf(2),cf(3) holds the coefficients for the polynom. at order c,b,a
        */

      int v1 = F(f,0); int v2 = F(f,1); int v3 = F(f,2);
      // get quadratic coefficients (ax^2 + b^x + c)
      const double& U11 = uv(v1,0);
      const double& U12 = uv(v1,1);
      const double& U21 = uv(v2,0);
      const double& U22 = uv(v2,1);
      const double& U31 = uv(v3,0);
      const double& U32 = uv(v3,1);

      const double& V11 = d(v1,0);
      const double& V12 = d(v1,1);
      const double& V21 = d(v2,0);
      const double& V22 = d(v2,1);
      const double& V31 = d(v3,0);
      const double& V32 = d(v3,1);

      double a = V11*V22 - V12*V21 - V11*V32 + V12*V31 + V21*V32 - V22*V31;
      double b = U11*V22 - U12*V21 - U21*V12 + U22*V11 - U11*V32 + U12*V31 + U31*V12 - U32*V11 + U21*V32 - U22*V31 - U31*V22 + U32*V21;
      double c = U11*U22 - U12*U21 - U11*U32 + U12*U31 + U21*U32 - U22*U31;

      return get_smallest_pos_quad_zero(a,b,c);
    }

    template<class Derived1, class Derived2>
    inline double get_min_pos_root_3D(const Eigen::MatrixBase<Derived1> &uv,
                                      const Eigen::MatrixXi &F,
                                      const Eigen::MatrixBase<Derived2> &d_, int f)
    {
      /*
          Searching for the roots of:
            +-1/6 * |ax ay az 1|
                    |bx by bz 1|
                    |cx cy cz 1|
                    |dx dy dz 1|
          Every point ax,ay,az has a search direction a_dx,a_dy,a_dz, and so we add those to the matrix, and solve the cubic to find the step size t for a 0 volume
          Symbolic matlab:
            syms a_x a_y a_z a_dx a_dy a_dz % tetrahedera point and search direction
            syms b_x b_y b_z b_dx b_dy b_dz
            syms c_x c_y c_z c_dx c_dy c_dz
            syms d_x d_y d_z d_dx d_dy d_dz
            syms t % Timestep var, this is what we're looking for


            a_plus_t = [a_x,a_y,a_z] + t*[a_dx,a_dy,a_dz];
            b_plus_t = [b_x,b_y,b_z] + t*[b_dx,b_dy,b_dz];
            c_plus_t = [c_x,c_y,c_z] + t*[c_dx,c_dy,c_dz];
            d_plus_t = [d_x,d_y,d_z] + t*[d_dx,d_dy,d_dz];

            vol_mat = [a_plus_t,1;b_plus_t,1;c_plus_t,1;d_plus_t,1]
            //cf = coeffs(det(vol_det),t); % Now cf(1),cf(2),cf(3),cf(4) holds the coefficients for the polynom
            [coefficients,terms] = coeffs(det(vol_det),t); % terms = [ t^3, t^2, t, 1], Coefficients hold the coeff we seek
      */
      int v1 = F(f,0); int v2 = F(f,1); int v3 = F(f,2); int v4 = F(f,3);
      const double& a_x = uv(v1,0);
      const double& a_y = uv(v1,1);
      const double& a_z = uv(v1,2);
      const double& b_x = uv(v2,0);
      const double& b_y = uv(v2,1);
      const double& b_z = uv(v2,2);
      const double& c_x = uv(v3,0);
      const double& c_y = uv(v3,1);
      const double& c_z = uv(v3,2);
      const double& d_x = uv(v4,0);
      const double& d_y = uv(v4,1);
      const double& d_z = uv(v4,2);

      const double& a_dx = d_(v1,0);
      const double& a_dy = d_(v1,1);
      const double& a_dz = d_(v1,2);
      const double& b_dx = d_(v2,0);
      const double& b_dy = d_(v2,1);
      const double& b_dz = d_(v2,2);
      const double& c_dx = d_(v3,0);
      const double& c_dy = d_(v3,1);
      const double& c_dz = d_(v3,2);
      const double& d_dx = d_(v4,0);
      const double& d_dy = d_(v4,1);
      const double& d_dz = d_(v4,2);

      // Find solution for: a*t^3 + b*t^2 + c*d +d = 0
      double a = a_dx*b_dy*c_dz - a_dx*b_dz*c_dy - a_dy*b_dx*c_dz + a_dy*b_dz*c_dx + a_dz*b_dx*c_dy - a_dz*b_dy*c_dx - a_dx*b_dy*d_dz + a_dx*b_dz*d_dy + a_dy*b_dx*d_dz - a_dy*b_dz*d_dx - a_dz*b_dx*d_dy + a_dz*b_dy*d_dx + a_dx*c_dy*d_dz - a_dx*c_dz*d_dy - a_dy*c_dx*d_dz + a_dy*c_dz*d_dx + a_dz*c_dx*d_dy - a_dz*c_dy*d_dx - b_dx*c_dy*d_dz + b_dx*c_dz*d_dy + b_dy*c_dx*d_dz - b_dy*c_dz*d_dx - b_dz*c_dx*d_dy + b_dz*c_dy*d_dx;

      double b = a_dy*b_dz*c_x - a_dy*b_x*c_dz - a_dz*b_dy*c_x + a_dz*b_x*c_dy + a_x*b_dy*c_dz - a_x*b_dz*c_dy - a_dx*b_dz*c_y + a_dx*b_y*c_dz + a_dz*b_dx*c_y - a_dz*b_y*c_dx - a_y*b_dx*c_dz + a_y*b_dz*c_dx + a_dx*b_dy*c_z - a_dx*b_z*c_dy - a_dy*b_dx*c_z + a_dy*b_z*c_dx + a_z*b_dx*c_dy - a_z*b_dy*c_dx - a_dy*b_dz*d_x + a_dy*b_x*d_dz + a_dz*b_dy*d_x - a_dz*b_x*d_dy - a_x*b_dy*d_dz + a_x*b_dz*d_dy + a_dx*b_dz*d_y - a_dx*b_y*d_dz - a_dz*b_dx*d_y + a_dz*b_y*d_dx + a_y*b_dx*d_dz - a_y*b_dz*d_dx - a_dx*b_dy*d_z + a_dx*b_z*d_dy + a_dy*b_dx*d_z - a_dy*b_z*d_dx - a_z*b_dx*d_dy + a_z*b_dy*d_dx + a_dy*c_dz*d_x - a_dy*c_x*d_dz - a_dz*c_dy*d_x + a_dz*c_x*d_dy + a_x*c_dy*d_dz - a_x*c_dz*d_dy - a_dx*c_dz*d_y + a_dx*c_y*d_dz + a_dz*c_dx*d_y - a_dz*c_y*d_dx - a_y*c_dx*d_dz + a_y*c_dz*d_dx + a_dx*c_dy*d_z - a_dx*c_z*d_dy - a_dy*c_dx*d_z + a_dy*c_z*d_dx + a_z*c_dx*d_dy - a_z*c_dy*d_dx - b_dy*c_dz*d_x + b_dy*c_x*d_dz + b_dz*c_dy*d_x - b_dz*c_x*d_dy - b_x*c_dy*d_dz + b_x*c_dz*d_dy + b_dx*c_dz*d_y - b_dx*c_y*d_dz - b_dz*c_dx*d_y + b_dz*c_y*d_dx + b_y*c_dx*d_dz - b_y*c_dz*d_dx - b_dx*c_dy*d_z + b_dx*c_z*d_dy + b_dy*c_dx*d_z - b_dy*c_z*d_dx - b_z*c_dx*d_dy + b_z*c_dy*d_dx;

      double c = a_dz*b_x*c_y - a_dz*b_y*c_x - a_x*b_dz*c_y + a_x*b_y*c_dz + a_y*b_dz*c_x - a_y*b_x*c_dz - a_dy*b_x*c_z + a_dy*b_z*c_x + a_x*b_dy*c_z - a_x*b_z*c_dy - a_z*b_dy*c_x + a_z*b_x*c_dy + a_dx*b_y*c_z - a_dx*b_z*c_y - a_y*b_dx*c_z + a_y*b_z*c_dx + a_z*b_dx*c_y - a_z*b_y*c_dx - a_dz*b_x*d_y + a_dz*b_y*d_x + a_x*b_dz*d_y - a_x*b_y*d_dz - a_y*b_dz*d_x + a_y*b_x*d_dz + a_dy*b_x*d_z - a_dy*b_z*d_x - a_x*b_dy*d_z + a_x*b_z*d_dy + a_z*b_dy*d_x - a_z*b_x*d_dy - a_dx*b_y*d_z + a_dx*b_z*d_y + a_y*b_dx*d_z - a_y*b_z*d_dx - a_z*b_dx*d_y + a_z*b_y*d_dx + a_dz*c_x*d_y - a_dz*c_y*d_x - a_x*c_dz*d_y + a_x*c_y*d_dz + a_y*c_dz*d_x - a_y*c_x*d_dz - a_dy*c_x*d_z + a_dy*c_z*d_x + a_x*c_dy*d_z - a_x*c_z*d_dy - a_z*c_dy*d_x + a_z*c_x*d_dy + a_dx*c_y*d_z - a_dx*c_z*d_y - a_y*c_dx*d_z + a_y*c_z*d_dx + a_z*c_dx*d_y - a_z*c_y*d_dx - b_dz*c_x*d_y + b_dz*c_y*d_x + b_x*c_dz*d_y - b_x*c_y*d_dz - b_y*c_dz*d_x + b_y*c_x*d_dz + b_dy*c_x*d_z - b_dy*c_z*d_x - b_x*c_dy*d_z + b_x*c_z*d_dy + b_z*c_dy*d_x - b_z*c_x*d_dy - b_dx*c_y*d_z + b_dx*c_z*d_y + b_y*c_dx*d_z - b_y*c_z*d_dx - b_z*c_dx*d_y + b_z*c_y*d_dx;

      double d = a_x*b_y*c_z - a_x*b_z*c_y - a_y*b_x*c_z + a_y*b_z*c_x + a_z*b_x*c_y - a_z*b_y*c_x - a_x*b_y*d_z + a_x*b_z*d_y + a_y*b_x*d_z - a_y*b_z*d_x - a_z*b_x*d_y + a_z*b_y*d_x + a_x*c_y*d_z - a_x*c_z*d_y - a_y*c_x*d_z + a_y*c_z*d_x + a_z*c_x*d_y - a_z*c_y*d_x - b_x*c_y*d_z + b_x*c_z*d_y + b_y*c_x*d_z - b_y*c_z*d_x - b_z*c_x*d_y + b_z*c_y*d_x;

      if (std::abs(a)<=1.e-10)
      {
        return get_smallest_pos_quad_zero(b,c,d);
      }
      b/=a; c/=a; d/=a; // normalize it all
      std::vector<double> res(3);
      int real_roots_num = SolveP3(res,b,c,d);
      switch (real_roots_num)
      {
        case 1:
          return (res[0] >= 0) ? res[0]:INFINITY;
        case 2:
        {
          double max_root = std::max(res[0],res[1]); double min_root = std::min(res[0],res[1]);
          if (min_root > 0) return min_root;
          if (max_root > 0) return max_root;
          return INFINITY;
        }
        case 3:
        default:
        {
          std::sort(res.begin(),res.end());
          if (res[0] > 0) return res[0];
          if (res[1] > 0) return res[1];
          if (res[2] > 0) return res[2];
          return INFINITY;
        }
      }
    }

    template<class Derived1, class Derived2>
    inline double compute_max_step_from_singularities(const Eigen::MatrixBase<Derived1> &uv,
                                                      const Eigen::MatrixXi &F,
                                                      const Eigen::MatrixBase<Derived2> &d)
    {
      double max_step = INFINITY;

      // The if statement is outside the for loops to avoid branching/ease parallelizing
      if (uv.cols() == 2)
      {
        max_step = tbb::parallel_reduce(tbb::blocked_range<int>(0, F.rows()), INFINITY,
            [&](const tbb::blocked_range<int> &r, double local_max_step) {
                for (int f = r.begin(); f < r.end(); f++) {
                    double min_positive_root = get_min_pos_root_2D(uv,F,d,f);
                    local_max_step = std::min(local_max_step, min_positive_root);
                }
                return local_max_step;
            },
            [](double a, double b) { return std::min(a, b); });
      }
      else
      { // volumetric deformation
        max_step = tbb::parallel_reduce(tbb::blocked_range<int>(0, F.rows()), INFINITY,
            [&](const tbb::blocked_range<int> &r, double local_max_step) {
                for (int f = r.begin(); f < r.end(); f++) {
                    double min_positive_root = get_min_pos_root_3D(uv,F,d,f);
                    local_max_step = std::min(local_max_step, min_positive_root);
                }
                return local_max_step;
            },
            [](double a, double b) { return std::min(a, b); });
      }
      return max_step;
    }
}

struct FlipAvoidingStepLength : public FeasibleStepLengthComputer {
    FlipAvoidingStepLength(const Eigen::MatrixXi &F)
        : F(F) {
        dim = F.cols() - 1;
        if (dim != 2 && dim != 3) {
            throw std::runtime_error("FlipAvoidingStepLength only supports 2D and 3D meshes, but got dimension " + std::to_string(dim) + ".");
        }
    }

    virtual Real eval(const VXd &vars, const VXd &step) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("FlipAvoidingStepLength.eval");
        int numNodes = vars.rows() / dim;
        if ((vars.rows() % dim) != 0) throw std::runtime_error("vars size must be a multiple of " + std::to_string(dim) + ", but is " + std::to_string(vars.rows()) + ".");
        using MXdr = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::Map<const MXdr> x(vars.data(), numNodes, dim);
        Eigen::MatrixXd d = Eigen::Map<const MXdr>(step.data(), numNodes, dim);
        return backoffFactor * flip_avoiding::compute_max_step_from_singularities(x, F, d);
    }

    Real backoffFactor = 0.8;
    int dim = 0;
    Eigen::MatrixXi F;
};

#endif /* end of include guard: FLIPAVOIDINGSTEPLENGTH_HH */
