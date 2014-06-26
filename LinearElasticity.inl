namespace LinearElasticity3D {
    template<class t_ETensorGetter>
    struct TetData : public LinearFEM3D::TetData {
        typedef LinearFEM3D::TetData Base;

        typedef Eigen::Matrix<Real,  3,  4> ElementLoad;
        typedef Eigen::Matrix<Real,  6, 12> PerElementSB;
        typedef Eigen::Matrix<Real, 12, 12> PerElementStiffness;

        TetData() { }

        const ETensor &E() const { return m_E(); }

        template<class FlattenedType>
        FlattenedType applyD(const FlattenedType  &in) const { return E().applyD(in); }

        template<class _SymMat>
        _SymMat applyE(const _SymMat &in) const { return _SymMat(E().doubleContract(in)); }

        template<class _ElemHandle, class _SymMat>
        void engStrain(_ElemHandle elem, const VField &u, _SymMat &&out) const {
            out.clear();
            for (size_t c = 0; c < elem.numVertices(); ++c) {
                const auto &uc = u(elem.vertex(c).index());
                out(0, 0) += m_gradPhis(c, 0) * uc[0];
                out(1, 1) += m_gradPhis(c, 1) * uc[1];
                out(2, 2) += m_gradPhis(c, 2) * uc[2];
                out(1, 2) += m_gradPhis(c, 1) * uc[2] + m_gradPhis(c, 2) * uc[1];
                out(0, 2) += m_gradPhis(c, 0) * uc[2] + m_gradPhis(c, 2) * uc[0];
                out(0, 1) += m_gradPhis(c, 1) * uc[0] + m_gradPhis(c, 0) * uc[1];
            }
        }

        template<class _ElemHandle, class _SymMat>
        void strain(_ElemHandle elem, const VField &u, _SymMat &&out) const {
            engStrain(elem, u, out);
            out(1, 2) /= 2; out(0, 2) /= 2; out(0, 1) /= 2;
        }

        template<class _ElemHandle, class _SymMat>
        void stress(_ElemHandle elem, const VField &u, _SymMat &&out) const {
            SMatrix smat;
            engStrain(elem, u, smat);
            out = applyD(smat.flattened());
        }

        // Load that a particular strain on this element puts on its nodes.
        // Effectively applies vol * B_e^t S D_e S.
        template<class _SymMat>
        void load(const _SymMat &strain, ElementLoad &l) const {
            SMatrix s = applyE(strain);
            s *= Base::volume();
            for (size_t c = 0; c < 4; ++c) {
                //       0     1     2     3     4     5
                // s: [s_xx, s_yy, s_zz, s_yz, s_xz, s_xy]
                l(0, c) = m_gradPhis(c, 0) * s[0] + m_gradPhis(c, 2) * s[4] + m_gradPhis(c, 1) * s[5]; // xx xz xy
                l(1, c) = m_gradPhis(c, 1) * s[1] + m_gradPhis(c, 2) * s[3] + m_gradPhis(c, 0) * s[5]; // yy yz yx
                l(2, c) = m_gradPhis(c, 2) * s[2] + m_gradPhis(c, 1) * s[3] + m_gradPhis(c, 0) * s[4]; // zz zy zx
            }
        }

        // Matrix computing engineering strain from corner displacements
        void perElementSB(Eigen::Matrix<Real, 6, 12> &SBe) const {
            SBe = PerElementSB::Zero();
            for (int c = 0; c < 4; ++c) {
                SBe(0, 3 * c + 0) = m_gradPhis(c, 0); // xx
                SBe(1, 3 * c + 1) = m_gradPhis(c, 1); // yy
                SBe(2, 3 * c + 2) = m_gradPhis(c, 2); // zz
                SBe(3, 3 * c + 1) = m_gradPhis(c, 2); SBe(3, 3 * c + 2) = m_gradPhis(c, 1); // yz + zy
                SBe(4, 3 * c + 2) = m_gradPhis(c, 0); SBe(4, 3 * c + 0) = m_gradPhis(c, 2); // zx + xz
                SBe(5, 3 * c + 0) = m_gradPhis(c, 1); SBe(5, 3 * c + 1) = m_gradPhis(c, 0); // xy + zx
            }
        }

        // Per-element stiffness matrix computing load from corner
        // displacements.
        void perElementStiffness(Eigen::Matrix<Real, 12, 12> &Ke) const {
            PerElementSB SBe;
            perElementSB(SBe);
            Ke = volume() * (SBe.transpose() * applyD(SBe));
        }

    protected:
        using Base::m_gradPhis;
        t_ETensorGetter m_E;
    };
}

namespace LinearElasticity2D {

}
