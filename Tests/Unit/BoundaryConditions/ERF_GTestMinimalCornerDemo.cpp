#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>

#include <ERF_Advection.H>

#include <gtest/gtest.h>
#include <iostream>

using namespace amrex;

namespace {

// The absolute minimum: one call, one obviously-wrong number. u=1 everywhere
// real (a boring, ordinary uniform flow -- RHS should be ~0), garbage in the
// one ghost cell nothing should read. dir=1 => this is the y-boundary
// tangent-x-momentum function; i=3 is the corner where x is ALSO Open.
TEST(MinimalCornerDemo, PrintTheCorruptedValue)
{
    const Box grown(IntVect(-2, -2, -2), IntVect(5, 2, 2));
    const Box bxx_with_corner(IntVect(0, 0, 0), IntVect(3, 0, 0));

    FArrayBox rho_u_rhs_fab(grown, 1), u_fab(grown, 1), rho_u_fab(grown, 1),
              rho_v_fab(grown, 1), omega_fab(grown, 1), ax_fab(grown, 1),
              az_fab(grown, 1), detJ_fab(grown, 1);
    rho_u_rhs_fab.setVal<RunOn::Host>(Real(-999.0));
    u_fab.setVal<RunOn::Host>(Real(1.0));         // ordinary uniform flow
    rho_u_fab.setVal<RunOn::Host>(Real(1.0));
    rho_v_fab.setVal<RunOn::Host>(Real(1.0));
    omega_fab.setVal<RunOn::Host>(Real(0.0));
    ax_fab.setVal<RunOn::Host>(Real(1.0));
    az_fab.setVal<RunOn::Host>(Real(1.0));
    detJ_fab.setVal<RunOn::Host>(Real(1.0));

    // The one out-of-domain ghost cell (i=4) this function has no business
    // reading. Nothing physical -- just an obviously wrong number.
    u_fab.array()(4, 0, 0, 0)     = Real(1.0e6);
    rho_u_fab.array()(4, 0, 0, 0) = Real(1.0e6);

    const GpuArray<Real, AMREX_SPACEDIM> cellSizeInv{Real(1.0), Real(1.0), Real(1.0)};

    AdvectionSrcForOpenBC_Tangent_Xmom(
        bxx_with_corner, /*dir=*/1, rho_u_rhs_fab.array(), u_fab.array(), rho_u_fab.array(),
        rho_v_fab.array(), omega_fab.array(), ax_fab.array(), az_fab.array(),
        detJ_fab.array(), cellSizeInv, /*do_lo=*/true);

    Real interior_value = rho_u_rhs_fab(IntVect(1, 0, 0)); // an ordinary, non-corner cell
    Real corner_value   = rho_u_rhs_fab(IntVect(3, 0, 0)); // the corner

    std::cout << "\n  ordinary in-domain cell (i=1) RHS: " << interior_value
              << "\n  corner cell            (i=3) RHS: " << corner_value
              << "\n  (uniform u=1 flow -- both 'should' be near zero)\n\n";

    EXPECT_NEAR(interior_value, Real(0.0), Real(1.0e-6));
}

} // namespace
