# E1 expert-labelled LC--MS peak-shape confirmation

Protocol: `docs/lcms_eic_e1_protocol.md` (frozen before data inspection).

Source: Müller et al. (2020), DOI `10.3390/metabo10040162`; Zenodo record
`10.5281/zenodo.3756211`, version 3. The source data are not redistributed.

## Locked design

- Double group holdout: sample and candidate-ion IDs are hashed independently;
  a case is used only when both axes agree on train, validation, or confirmation.
- Confirmation contains 2692 unambiguous EICs (297 expert peaks and 2395
  expert non-peaks), eight sample groups, and 706 candidate-ion groups.
- One fixed histogram-gradient-boosted learner is used for every
  representation. Validation selected `DOMAIN` as the non-HCRD comparator.
- Primary endpoint: AP difference `HCRD-8 - DOMAIN`; 10,000-replicate two-way
  cluster bootstrap over samples and candidate ions.

## Locked result

| Representation | AP | ROC AUC | MCC | Balanced accuracy |
|---|---:|---:|---:|---:|
| RAW64 | 0.362590 | 0.871413 | 0.415022 | 0.778237 |
| DOMAIN | 0.381973 | 0.880109 | 0.437874 | 0.786588 |
| HCRD-1 | 0.377139 | 0.869620 | 0.402628 | 0.759429 |
| HCRD-8 | **0.409190** | **0.884417** | **0.441780** | **0.813306** |
| HCRD-GEOMETRY | 0.334511 | 0.841517 | 0.347912 | 0.740902 |
| AREA-ONLY | 0.315068 | 0.825381 | 0.336861 | 0.745069 |

`HCRD-8 - DOMAIN` AP = `+0.027217`, two-way cluster 95% CI
`[-0.056573, 0.101046]`, two-sided bootstrap `p=0.51695`. HCRD-8 exceeded
HCRD-1, but the interval criterion failed, so prospective success is **false**.

## Interpretation

This is positive mechanism evidence, not statistical confirmation of
superiority. It shows that the full hierarchy can add signal beyond one level
on untouched expert-labelled real EICs. Area-only and scalar-geometry
ablations do not carry that result: the useful representation includes the
signed detail waveforms, raw local shape, and multilevel structural summaries.
The wide primary interval is driven by the intentionally conservative double
holdout, especially its eight independent confirmation samples.
