# Contributing

Bug reports, counterexamples, independent replications, and comparisons with
closely related methods are especially valuable.

Before submitting a code change:

1. install `.[dev,comparisons]`;
2. run `python -m pytest -q`;
3. add a regression test for changed mathematical behavior;
4. update the relevant theorem or experiment documentation.

New confirmatory experiments must have a written protocol, fixed seed family,
primary endpoint, independent inference unit, and success criterion before the
new outcomes are inspected.  Exploratory work is welcome but must remain
labelled exploratory.

Do not add third-party raw datasets to the repository.  Add source URLs,
licenses/citations, checksums, and deterministic download/preparation code.
