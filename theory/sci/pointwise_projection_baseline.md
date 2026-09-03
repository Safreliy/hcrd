# Pointwise-band projection: an honest matched baseline

## Construction

Let `Y_i = mu_i + sigma Z_i` on a fixed ordered design, with standard Gaussian
errors and known `sigma`. Build the simultaneous pointwise intervals

`Y_i +/- z_(1-alpha/(2n)) sigma`.

For every cut between consecutive design points, ask whether some vector
inside this confidence box has nondecreasing secant slopes before the cut and
nonincreasing secant slopes after it. This is a linear feasibility problem.
Keep every feasible cut and return their range. The implementation also keeps
the two boundary cuts, which represent a fully concave or fully convex curve.

## Finite-sample guarantee

With probability at least `1-alpha`, every sampled true mean `mu_i` lies in its
pointwise interval. If the underlying function has a convex-to-concave
transition at `m`, choose the cut immediately before `m`. The sampled true mean
is discretely convex before that cut and discretely concave after it. It is
therefore a feasible vector, so the projected set contains `m`.

This proof allows a nonsmooth or discontinuous transition. It is a confidence
statement about the sampled shape and uses a more generic confidence region
than SCI.

## Computational shortcut

Feasibility of convex prefixes can only change from true to false as the
prefix grows. Feasibility of concave suffixes can only change from false to
true as the starting index moves right. Two binary searches therefore find
the full interval of feasible cuts with `O(log n)` linear programs rather than
one program for every cut.

## Role in the paper

This method is called pointwise-band projection (PBP). It is our matched honest
baseline. It is related to the confidence-region strategy of Davies, Kovac and
Meise (2009), but it is not their official procedure and does not use their
multiscale interval region.
