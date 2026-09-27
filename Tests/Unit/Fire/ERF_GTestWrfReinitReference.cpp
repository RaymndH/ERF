#include <gtest/gtest.h>
#include <AMReX_REAL.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <cmath>
#include <vector>
#include <cstdio>

#include "ERF_WrfReinitReference.H"

/**
 * @file ERF_GTestWrfReinitReference.cpp
 * @brief Sanity check for ERF_WrfReinitReference.H (the exact C++ mirror of
 * WRF-Fire's reinit_ls_rk3/advance_ls_reinit), run through the SAME kind of
 * static, no-advection front-displacement measurement as T1
 * (ERF_GTestReinitDrift.cpp), so WRF's own algorithm's behavior on a flat
 * front can be compared directly against ERF's under identical conditions.
 *
 * Domain/shape setup deliberately mirrors T1's (5000m square, dx=50m,
 * sub-cell-offset straight front) so results are apples-to-apples.
 */

using namespace amrex;
using namespace wrf_reinit_reference;

namespace {

constexpr int  NCELL = 100;
constexpr Real LDOM  = 5000.0;
constexpr Real DX    = LDOM / NCELL;
constexpr Real CX    = 0.5 * LDOM;
constexpr Real CY    = 0.5 * LDOM;
constexpr Real OFFX  = 0.37 * DX;
constexpr Real OFFY  = 0.61 * DX;

struct Grid
{
    BoxArray            ba;
    DistributionMapping dm;
    Geometry            geom;
    Grid ()
    {
        Box domain(IntVect(0, 0, 0), IntVect(NCELL - 1, NCELL - 1, 0));
        ba = BoxArray(domain);
        dm = DistributionMapping(ba);
        RealBox rb({0.0, 0.0, 0.0}, {LDOM, LDOM, 1.0});
        geom = Geometry(domain, rb, CoordSys::cartesian, {0, 0, 0});
    }
};

Real xc (int i) { return (i + Real(0.5)) * DX; }
Real yc (int j) { return (j + Real(0.5)) * DX; }

struct LineShape
{
    Real nx, ny, x0, y0;
    explicit LineShape (Real deg)
    {
        const Real t = deg * M_PI / 180.0;
        nx = std::cos(t); ny = std::sin(t);
        x0 = CX + OFFX; y0 = CY + OFFY;
    }
    Real phi (Real x, Real y) const { return (x - x0) * nx + (y - y0) * ny; }
    std::vector<std::pair<Real,Real>> front_points (int n = 50) const
    {
        std::vector<std::pair<Real,Real>> pts;
        const Real tx = -ny, ty = nx;
        const Real half_len = 0.35 * LDOM;
        for (int k = 0; k < n; ++k) {
            const Real s = -half_len + 2.0 * half_len * k / (n - 1);
            pts.emplace_back(x0 + s * tx, y0 + s * ty);
        }
        return pts;
    }
};

// Bilinear interpolation of a single-box, cell-centred MultiFab.
Real interp (const MultiFab& mf, Real x, Real y)
{
    Real fi = x / DX - 0.5, fj = y / DX - 0.5;
    int i0 = static_cast<int>(std::floor(fi));
    int j0 = static_cast<int>(std::floor(fj));
    i0 = amrex::max(0, amrex::min(NCELL - 2, i0));
    j0 = amrex::max(0, amrex::min(NCELL - 2, j0));
    const Real tx = amrex::max(Real(0.0), amrex::min(Real(1.0), fi - i0));
    const Real ty = amrex::max(Real(0.0), amrex::min(Real(1.0), fj - j0));
    const auto& fab = mf[0];
    const auto arr = fab.const_array();
    auto V = [&](int i, int j) { return arr(i, j, 0); };
    return (1-tx)*(1-ty)*V(i0,j0) + tx*(1-ty)*V(i0+1,j0)
         + (1-tx)*ty*V(i0,j0+1) + tx*ty*V(i0+1,j0+1);
}

void fill_exact (MultiFab& phi, const LineShape& shape)
{
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto arr = phi.array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            arr(i,j,k) = shape.phi(xc(i), yc(j));
        });
    }
}

// Mean/max outward-normal displacement of the front after N reinit calls,
// same first-order estimate T1 uses: delta(s) = -phi_measured(orig front pt).
std::pair<Real,Real> front_displacement (const MultiFab& phi, const LineShape& shape)
{
    Real sum = 0.0, maxabs = 0.0;
    auto pts = shape.front_points();
    for (auto& p : pts) {
        const Real d = -interp(phi, p.first, p.second);
        sum += d;
        maxabs = amrex::max(maxabs, std::abs(d));
    }
    return {sum / pts.size(), maxabs};
}

/// Circle of radius R0 about (CX+OFFX, CY+OFFY) -- same convention as T1's
/// CircleShape (ERF_GTestReinitDrift.cpp). convex=true: burned inside
/// (phi = r - R0); convex=false: burned outside (phi = R0 - r, concave).
struct CircleShape
{
    Real R0, cx, cy; bool convex;
    CircleShape (Real R, bool conv) : R0(R), cx(CX + OFFX), cy(CY + OFFY), convex(conv) {}
    Real phi (Real x, Real y) const
    {
        const Real r = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
        return convex ? (r - R0) : (R0 - r);
    }
    std::vector<std::pair<Real,Real>> front_points (int n = 50) const
    {
        std::vector<std::pair<Real,Real>> pts;
        for (int k = 0; k < n; ++k) {
            const Real a = 2.0 * M_PI * k / n;
            pts.emplace_back(cx + R0 * std::cos(a), cy + R0 * std::sin(a));
        }
        return pts;
    }
};

void fill_exact (MultiFab& phi, const CircleShape& shape)
{
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto arr = phi.array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            arr(i,j,k) = shape.phi(xc(i), yc(j));
        });
    }
}

template <class Shape>
std::pair<Real,Real> front_displacement_t (const MultiFab& phi, const Shape& shape)
{
    Real sum = 0.0, maxabs = 0.0;
    auto pts = shape.front_points();
    for (auto& p : pts) {
        const Real d = -interp(phi, p.first, p.second);
        sum += d;
        maxabs = amrex::max(maxabs, std::abs(d));
    }
    return {sum / pts.size(), maxabs};
}

} // namespace

TEST(WrfReinitReference, CurvedFrontCreepConvexVsConcave)
{
    // Same R0=1000m, same domain/dx, same N checkpoints as T1
    // (ReinitDrift.StaticFrontDisplacement's circle_convex_R1000 /
    // circle_concave_R1000 cases), so results are directly comparable.
    // WRF's own algorithm always applies its final min-clamp (there is no
    // "unclamped" mode in WRF-Fire) -- fire_lsm_reinit_iter=1 (the campaign
    // namelist default) is used as n_outer, called N times to match "N
    // reinit calls" the same way T1 counts them for ERF.
    Grid grid;
    for (bool convex : {true, false}) {
        CircleShape shape(1000.0, convex);
        MultiFab phi(grid.ba, grid.dm, 1, 3);
        fill_exact(phi, shape);

        int done = 0;
        MultiFab work(grid.ba, grid.dm, 1, 3);
        work.ParallelCopy(phi);
        for (int cp : {1, 10, 100, 1000}) {
            for (int c = 0; c < cp - done; ++c) {
                wrf_reinit_ls_rk3(work, grid.geom, /*n_outer=*/1, /*band_ngp=*/4.0);
            }
            done = cp;
            auto [mean_d, max_d] = front_displacement_t(work, shape);
            std::printf("[WrfReinitReference] shape=circle_%s_R1000 N=%4d  "
                        "mean_disp=% .6f dx  max_disp=%.6f dx\n",
                        convex ? "convex" : "concave", cp, mean_d / DX, max_d / DX);
        }
    }
}

TEST(WrfReinitReference, FlatFrontNoDriftAtObliqueAngles)
{
    Grid grid;
    for (Real angle : {0.0, 22.5, 34.0}) {
        LineShape shape(angle);
        MultiFab phi(grid.ba, grid.dm, 1, 3);
        fill_exact(phi, shape);

        for (int N : {1, 10, 100}) {
            MultiFab work(grid.ba, grid.dm, 1, 3);
            work.ParallelCopy(phi);
            for (int call = 0; call < N; ++call) {
                wrf_reinit_ls_rk3(work, grid.geom, /*n_outer=*/1, /*band_ngp=*/4.0);
            }
            auto [mean_d, max_d] = front_displacement(work, shape);
            std::printf("[WrfReinitReference] angle=%.1fdeg N=%4d  mean_disp=%.6f dx  max_disp=%.6f dx\n",
                        angle, N, mean_d / DX, max_d / DX);
            // Same expectation T1 confirmed for ERF: a flat front's crossing
            // should not move at any iteration count, clamp, or angle.
            EXPECT_LT(std::abs(mean_d) / DX, 0.05)
                << "angle=" << angle << " N=" << N;
        }
    }
}
