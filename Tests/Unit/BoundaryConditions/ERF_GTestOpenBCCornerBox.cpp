#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>

#include <ERF_Advection.H>

#include <gtest/gtest.h>

using amrex::Array4;
using amrex::Box;
using amrex::FArrayBox;
using amrex::GpuArray;
using amrex::IntVect;
using amrex::Real;

namespace {

// A small, generic domain (cells 0..9, 0..9, 0) with two perpendicular
// Open faces -- the geometry-independent minimum needed to reproduce this
// bug. Any domain with two adjacent Open lateral faces has this corner
// regardless of what terrain or flow is or isn't present near it.
Box test_domain () { return Box(IntVect(0, 0, 0), IntVect(9, 9, 0)); }

// Motivation: with neither adjacent face open, the tangential box used for
// a face that ISN'T at a corner (or whose corner is owned elsewhere) must be
// returned completely unchanged -- shrinking is opt-in per open face, never
// unconditional.
TEST(OpenBCCornerBox, NoAdjacentOpenFaceLeavesBoxUnchanged)
{
    const Box original(IntVect(0, 0, 0), IntVect(0, 9, 0));
    const Box shrunk = ShrinkTangentBoxForOpenBCCorner(original, /*dim=*/1,
                                                        /*lo_open=*/false, /*hi_open=*/false);
    EXPECT_EQ(shrunk, original);
}

// Motivation: an open low face in the shrink dimension must remove exactly
// the one corner cell at that end, not more (over-shrinking would silently
// drop legitimate non-corner cells from the tangential treatment) and not
// zero (which is the original double-write bug).
TEST(OpenBCCornerBox, OpenLowFaceShrinksExactlyOneCellAtLowEnd)
{
    const Box original(IntVect(0, 0, 0), IntVect(0, 9, 0));
    const Box shrunk = ShrinkTangentBoxForOpenBCCorner(original, /*dim=*/1,
                                                        /*lo_open=*/true, /*hi_open=*/false);
    EXPECT_EQ(shrunk.smallEnd(1), original.smallEnd(1) + 1);
    EXPECT_EQ(shrunk.bigEnd(1), original.bigEnd(1));
    EXPECT_EQ(shrunk.length(1), original.length(1) - 1);
}

// Motivation: symmetric case for the high end, and both ends open at once
// (all four lateral faces Open, as in the case that found this bug) must
// shrink both ends independently rather than one clobbering the other.
TEST(OpenBCCornerBox, OpenHighFaceAndBothFacesShrinkIndependently)
{
    const Box original(IntVect(0, 0, 0), IntVect(0, 9, 0));

    const Box hi_only = ShrinkTangentBoxForOpenBCCorner(original, 1, false, true);
    EXPECT_EQ(hi_only.smallEnd(1), original.smallEnd(1));
    EXPECT_EQ(hi_only.bigEnd(1), original.bigEnd(1) - 1);

    const Box both = ShrinkTangentBoxForOpenBCCorner(original, 1, true, true);
    EXPECT_EQ(both.smallEnd(1), original.smallEnd(1) + 1);
    EXPECT_EQ(both.bigEnd(1), original.bigEnd(1) - 1);
    EXPECT_EQ(both.length(1), original.length(1) - 2);
}

// Motivation: a box that is already empty (e.g. this MPI rank's tile
// doesn't touch this domain face at all) must stay empty and not be
// resurrected into a bogus non-empty box by the shrink -- Box::growLo/growHi
// on an invalid box is undefined territory this helper must avoid entering.
TEST(OpenBCCornerBox, EmptyBoxStaysEmpty)
{
    const Box empty;
    ASSERT_FALSE(empty.ok());
    const Box shrunk = ShrinkTangentBoxForOpenBCCorner(empty, 0, true, true);
    EXPECT_FALSE(shrunk.ok());
}

// Motivation: the actual bug this function fixes. At the (xhi, ylo) corner
// of a domain with Open BCs on both faces, the x-normal box (owning the x=9
// boundary column) and the y-tangential box (owning the y=0 boundary row,
// after this fix's shrink) must NOT both claim the corner cell (9,0,0) --
// that double-claim, with the tangential formula's out-of-domain read, was
// the root cause of the corner corruption. Before this fix (i.e., without
// calling ShrinkTangentBoxForOpenBCCorner at all), the two boxes DID both
// contain the corner cell; this test would have failed against the old code.
TEST(OpenBCCornerBox, NormalAndShrunkTangentBoxesDoNotShareTheCorner)
{
    const Box domain = test_domain();
    const IntVect corner(domain.bigEnd(0), domain.smallEnd(1), 0); // (xhi, ylo)

    // The x-normal box owning the xhi boundary column (a 1-cell-thick slab
    // at x=9, spanning all of y) -- mirrors makeSlab(tbx, 0, domain.bigEnd(0))
    // in AdvectionSrcForMom's xhi_open branch.
    const Box x_normal_box(IntVect(domain.bigEnd(0), domain.smallEnd(1), 0),
                            IntVect(domain.bigEnd(0), domain.bigEnd(1), 0));
    ASSERT_TRUE(x_normal_box.contains(corner));

    // The y-tangential box, before the fix, would be a 1-cell-thick slab at
    // y=0 spanning all of x -- also containing the corner. After the fix,
    // shrunk by the x-open flags exactly like the ylo_open branch does for
    // tbx_ylo/tbz_ylo in the real kernel.
    const Box y_tangent_box_unfixed(IntVect(domain.smallEnd(0), domain.smallEnd(1), 0),
                                     IntVect(domain.bigEnd(0), domain.smallEnd(1), 0));
    ASSERT_TRUE(y_tangent_box_unfixed.contains(corner))
        << "sanity check: the corner really was double-claimed before the fix";

    const bool xlo_open = true, xhi_open = true;
    const Box y_tangent_box_fixed = ShrinkTangentBoxForOpenBCCorner(
        y_tangent_box_unfixed, /*dim=*/0, xlo_open, xhi_open);

    EXPECT_FALSE(y_tangent_box_fixed.contains(corner))
        << "the shrunk tangential box must not overlap the corner the "
           "normal-direction box already owns -- if it does, the corner "
           "cell is written twice per step, reintroducing the bug";
}

// Motivation: the production kernel itself, not just its box geometry, must
// not read past-domain ghost data at a corner. This calls the real
// AdvectionSrcForOpenBC_Tangent_Xmom (used for the ylo/yhi_open branches'
// tbx_ylo/tbx_yhi boxes) directly on a hand-built field, on a small flat
// (no terrain, ax=az=detJ=1) domain with x ALSO open at its high end
// (xhi_open) -- so the corner cell (3,0,0) sits where this y-tangential
// box would, before the fix, overlap the x-normal box's territory.
//
// Rather than hand-deriving the "correct" flux value at the corner (there
// isn't a well-defined one -- that's the whole point: the corner belongs
// to the x-normal treatment, not this one), this proves corruption the
// more direct way: a physically meaningful result for an in-domain cell
// must not depend on what's sitting in the ghost cell one step outside
// the domain. Two runs that differ only in that single ghost value must
// give the SAME answer at (3,0,0) if the kernel is well-behaved there.
// Before the fix (unshrunk box), they don't. After the fix (shrunk box),
// the cell isn't touched by this function at all, so it trivially can't
// depend on the ghost -- confirmed by checking it stays at its untouched
// sentinel value regardless of the poison.
TEST(OpenBCCornerBox, TangentMomentumKernelCornerDoesNotDependOnGhostData)
{
    // Flat, uniform-grid domain: i=0..3, one row (j=0) at the ylo boundary,
    // one z-layer. xhi_open=true makes i=3 a corner column. Ghost padding
    // wide enough for every i/j/k +-1 access the kernel makes internally.
    const Box grown(IntVect(-2, -2, -2), IntVect(5, 2, 2));
    const Box bxx_unfixed(IntVect(0, 0, 0), IntVect(3, 0, 0)); // full row, incl. corner i=3

    auto build_fields = [&](Real poison_at_ghost, FArrayBox& rho_u_rhs_fab,
                             FArrayBox& u_fab, FArrayBox& rho_u_fab, FArrayBox& rho_v_fab,
                             FArrayBox& omega_fab, FArrayBox& ax_fab, FArrayBox& az_fab,
                             FArrayBox& detJ_fab, Real sentinel)
    {
        rho_u_rhs_fab.resize(grown, 1); rho_u_rhs_fab.setVal<amrex::RunOn::Host>(sentinel);
        u_fab.resize(grown, 1);
        rho_u_fab.resize(grown, 1);
        rho_v_fab.resize(grown, 1);     rho_v_fab.setVal<amrex::RunOn::Host>(Real(1.0));
        omega_fab.resize(grown, 1);     omega_fab.setVal<amrex::RunOn::Host>(Real(0.0));
        ax_fab.resize(grown, 1);        ax_fab.setVal<amrex::RunOn::Host>(Real(1.0));
        az_fab.resize(grown, 1);        az_fab.setVal<amrex::RunOn::Host>(Real(1.0));
        detJ_fab.resize(grown, 1);      detJ_fab.setVal<amrex::RunOn::Host>(Real(1.0));

        const auto u_arr = u_fab.array();
        const auto rho_u_arr = rho_u_fab.array();
        const Real U0 = Real(10.0), SLOPE = Real(0.5);
        const Real RHO0 = Real(1.2), RSLOPE = Real(0.01);
        amrex::LoopOnCpu(grown, [&](int i, int j, int k)
        {
            if (i <= 3) {
                u_arr(i, j, k, 0)     = U0 + SLOPE * Real(i);
                rho_u_arr(i, j, k, 0) = RHO0 + RSLOPE * Real(i);
            } else {
                // i == 4: exactly the ghost cell AdvectionSrcForOpenBC_Tangent_Xmom
                // reads via rho_u(i+1,...)/u(i+1,...) at the corner column i=3,
                // one step outside the domain (xhi boundary is at i=3). A
                // well-behaved boundary kernel must not depend on this value.
                u_arr(i, j, k, 0)     = poison_at_ghost;
                rho_u_arr(i, j, k, 0) = poison_at_ghost;
            }
        });
    };

    const bool xlo_open = false, xhi_open = true;
    const Box bxx_fixed = ShrinkTangentBoxForOpenBCCorner(bxx_unfixed, /*dim=*/0, xlo_open, xhi_open);
    ASSERT_FALSE(bxx_fixed.contains(IntVect(3, 0, 0)))
        << "sanity check: the fix must exclude the corner column from this box";

    const GpuArray<Real, AMREX_SPACEDIM> cellSizeInv{Real(1.0), Real(1.0), Real(1.0)};
    const Real sentinel = Real(-999.0);

    Real corner_result_unfixed[2];
    for (int trial = 0; trial < 2; ++trial) {
        const Real poison = (trial == 0) ? Real(1.0e6) : Real(-3.7e6);
        FArrayBox rho_u_rhs_fab, u_fab, rho_u_fab, rho_v_fab, omega_fab, ax_fab, az_fab, detJ_fab;
        build_fields(poison, rho_u_rhs_fab, u_fab, rho_u_fab, rho_v_fab,
                     omega_fab, ax_fab, az_fab, detJ_fab, sentinel);

        AdvectionSrcForOpenBC_Tangent_Xmom(
            bxx_unfixed, /*dir=*/1, rho_u_rhs_fab.array(), u_fab.array(), rho_u_fab.array(),
            rho_v_fab.array(), omega_fab.array(), ax_fab.array(), az_fab.array(),
            detJ_fab.array(), cellSizeInv, /*do_lo=*/true);

        corner_result_unfixed[trial] = rho_u_rhs_fab(IntVect(3, 0, 0));
    }
    EXPECT_NE(corner_result_unfixed[0], corner_result_unfixed[1])
        << "before the fix, the corner's result changed when only an out-of-domain "
           "ghost cell changed -- confirming it reads data it has no business reading";

    for (int trial = 0; trial < 2; ++trial) {
        const Real poison = (trial == 0) ? Real(1.0e6) : Real(-3.7e6);
        FArrayBox rho_u_rhs_fab, u_fab, rho_u_fab, rho_v_fab, omega_fab, ax_fab, az_fab, detJ_fab;
        build_fields(poison, rho_u_rhs_fab, u_fab, rho_u_fab, rho_v_fab,
                     omega_fab, ax_fab, az_fab, detJ_fab, sentinel);

        AdvectionSrcForOpenBC_Tangent_Xmom(
            bxx_fixed, /*dir=*/1, rho_u_rhs_fab.array(), u_fab.array(), rho_u_fab.array(),
            rho_v_fab.array(), omega_fab.array(), ax_fab.array(), az_fab.array(),
            detJ_fab.array(), cellSizeInv, /*do_lo=*/true);

        EXPECT_EQ(rho_u_rhs_fab(IntVect(3, 0, 0)), sentinel)
            << "after the fix, the corner cell must not be written by this function "
               "at all, regardless of what garbage sits in its ghost cell";
    }
}

} // namespace
