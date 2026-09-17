#include <AMReX_Box.H>

#include <ERF_Advection.H>

#include <gtest/gtest.h>

using amrex::Box;
using amrex::IntVect;

namespace {

// A small stand-in for a single 2D slab of a fire-relevant domain: cells
// (0..9, 0..9, 0), matching the kind of domain the corner bug was found on
// (a rotated ridge case with Open BCs on all four lateral faces).
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

} // namespace
