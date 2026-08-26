# D1 cross-source population screen: stopped before HCRD

D1 was designed after the successful Yahoo C1 confirmation to test the
unchanged `hcrd_L8_max` detector on a different official source family. The
protocol required at least ten previously unused KDD21/UCR point-anomaly
series and was written before population matching or HCRD execution.

The deterministic screen found 152 candidate KDD21/UCR series absent from both
the TSB-AD tuning and evaluation lists. Of these, 145 had a unique content match
and 7 were excluded as missing or nonunique; only **5** of the unique matches
carried the official `point_anom == 1` flag. The predeclared minimum was not
met. The runner therefore wrote a population audit and returned without
computing any HCRD score, selecting a comparator, or creating a frozen
evaluation configuration.

This is not a negative accuracy result; it is an insufficient-sample abort. It
prevents presenting five files as independent confirmation after the fact and
leaves the following gap explicit: a cross-source point-transient suite with
enough labelled series and strong rerunnable comparators is still required.

Artifacts:

- `docs/tsb_uad_kdd21_d1_protocol.md`: pre-execution minimum and matching rule;
- `results/tsb_uad_kdd21_d1/population_manifest.csv`: 145 unique matches;
- `results/tsb_uad_kdd21_d1/population_screen.json`: abort status and hashes;
- `results/tsb_uad_kdd21_d1/excluded_matches.csv`: unmatched/ambiguous records;
- `experiments/run_tsb_uad_kdd21_confirmation.py`: guarded runner; evaluation
  remains impossible because no frozen configuration exists.
