# E2 dependence, source-refit, and acquisition-file sensitivity

The fixed-source analysis resampled target RT blocks while preserving the original no-refit estimand. It used 10,000 paired replicates at each width. The source-refit analysis then refit the learner inside every RT-block replicate; its primary width used 1,000 replicates and the two width sensitivities used 300 each.

| Design | Falkor to MESOSCOPE | MESOSCOPE to Falkor |
|---|---:|---:|
| Fixed-source target RT blocks (30 s) | 0.1181 [0.0340, 0.2051] | 0.1005 [0.0402, 0.1762] |
| Fixed-source target RT blocks (60 s) | 0.1181 [0.0130, 0.2190] | 0.1005 [0.0338, 0.1965] |
| Fixed-source target RT blocks (120 s) | 0.1181 [-0.0227, 0.2418] | 0.1005 [0.0415, 0.2082] |
| Source-refit RT blocks (30 s) | 0.1087 [-0.0105, 0.2074] | 0.0556 [-0.0671, 0.1602] |
| Source-refit RT blocks (60 s) | 0.1000 [-0.0444, 0.2179] | 0.0515 [-0.0935, 0.1742] |
| Source-refit RT blocks (120 s) | 0.0886 [-0.1514, 0.2508] | 0.0574 [-0.0803, 0.1761] |
| Acquisition-file delete-group range | 0.1030 to 0.1397; 10/10 positive | 0.0517 to 0.1058; 10/10 positive |

At the primary 60-second width, both fixed-source target-block intervals remained above zero. Five of the six width-by-direction intervals were positive; the Falkor-to-MESOSCOPE interval crossed zero only at 120 seconds. This supports the stated conditional transfer estimand under local target dependence.

The source-refit mean was positive in every direction and block-width design. The fraction of positive paired replicates was 30 s: 96.3%/85.3%; 60 s: 92.9%/81.0%; 120 s: 87.3%/80.7% for Falkor-to-MESOSCOPE/MESOSCOPE-to-Falkor. All six percentile intervals nevertheless crossed zero, so this sensitivity supports a positive point effect but not a bootstrap sign claim after source refitting and RT-block resampling.

The fixed-source analysis isolates target dependence. The source-refit intervals additionally propagate source-model uncertainty, and deterministic file deletion repeats per-file aggregation and model fitting. None of the analyses can recover unavailable compound/adduct identifiers or represent additional laboratories.
