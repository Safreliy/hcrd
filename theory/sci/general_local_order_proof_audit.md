# Internal adversarial audit of THM E33.3

This is an internal proof audit, not an independent referee report.

## Round 1: does the theorem localize the right target?

The first draft used one point `m_0`. That would silently assume a unique
transition and would lose the identified-set contribution. The repaired
statement uses `I_f=[m_-,m_+]` and bounds only the excess length of the
confidence set beyond this interval.

## Round 2: are the support endpoints in the right direction?

A positive contrast gives a lower bound equal to its left support endpoint.
It is not enough for its right endpoint to be close to `m_-`; the left endpoint
must be close as well. The statement now assumes that the entire positive
support is left of `m_-` and that its left endpoint is at least `m_- - Kd`.
The negative condition is the symmetric statement using its right endpoint.

## Round 3: does certification require the simultaneous event?

No. The probability that the selected positive or negative row is certified
comes from its own Gaussian tail. The simultaneous event is needed for the
other direction: it prevents any additional certified row from removing a
true transition. The proof now separates these events and spends `eta/2`,
`eta/2`, and `alpha` before applying the union bound.

## Round 4: is the general rate claimed from shape alone?

No. Convexity and concavity determine contrast signs but do not give a lower
bound on their magnitudes. An affine curve is the immediate counterexample.
The theorem therefore states a `B d^gamma` margin assumption explicitly. It
also states the support-availability and coefficient-norm conditions needed
for irregular designs. Only after these conditions are given does the usual
`1/(2 gamma+1)` exponent follow.

## Remaining external obligation

The probability argument is short, but the result should still be checked by
an independent statistician before submission. A stronger paper would also
derive the margin and support conditions from a named function class and a
quasi-uniform or random-design assumption.
