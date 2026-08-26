# HCRD geometric-mass invariants

For one nonzero signed chord residual $d$ on an interval of duration $W$, set

\[
H=\|d\|_\infty,\quad A=\int |d|,\quad E=\int d^2,
\quad F=\frac{A}{WH},\quad C=\frac{A^2}{WE}.
\]

`F` is lobe fullness and `C` is quadratic concentration. The bounds
$0<F,C\le1$ follow respectively from $|d|\le H$ and Cauchy--Schwarz. Under
$d(x)\mapsto c\,d((x-\tau)/\rho)$, $A\mapsto |c|\rho A$,
$E\mapsto c^2\rho E$, $W\mapsto\rho W$, and $H\mapsto|c|H$, so both ratios are
invariant. Affine trend addition cancels before the residual is formed.

Two reference shapes are exact:

| Lobe | $F$ | $C$ |
|---|---:|---:|
| piecewise-linear triangle, arbitrary apex | $1/2$ | $3/4$ |
| parabola $4Ht(W-t)/W^2$ | $2/3$ | $5/6$ |

These descriptors explain the intended hierarchy of information:

1. polygon mass $A$ records size in physical units;
2. $(W,H)$ separates duration and amplitude;
3. $(F,C)$ records scale-free shape;
4. the signed detail samples retain morphology not captured by any finite
   scalar list;
5. the sequence across HCRD levels records evolution and persistence.

The inequalities and scaling identities are proofs. Their usefulness for a
labelled task is empirical and must be evaluated with an untouched split. The
LC--MS E1/E2 protocols were frozen before `C` was added, so it is not silently
inserted into those results.
