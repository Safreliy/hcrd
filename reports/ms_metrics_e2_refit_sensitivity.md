# E2 source-refit and acquisition-file sensitivity

The source learner was refit inside every RT-block bootstrap replicate; target RT blocks were resampled independently and paired across representations. The primary block width used 1,000 replicates, and the two width sensitivities used 300 each.

| Design | Falkor to MESOSCOPE | MESOSCOPE to Falkor |
|---|---:|---:|
| Source-refit RT blocks (30 s) | 0.1087 [-0.0105, 0.2074] | 0.0556 [-0.0671, 0.1602] |
| Source-refit RT blocks (60 s) | 0.1000 [-0.0444, 0.2179] | 0.0515 [-0.0935, 0.1742] |
| Source-refit RT blocks (120 s) | 0.0886 [-0.1514, 0.2508] | 0.0574 [-0.0803, 0.1761] |
| Acquisition-file delete-group range | 0.1030 to 0.1397; 10/10 positive | 0.0517 to 0.1058; 10/10 positive |

The source-refit mean was positive in every direction and block-width design. The fraction of positive paired replicates was 30 s: 96.3%/85.3%; 60 s: 92.9%/81.0%; 120 s: 87.3%/80.7% for Falkor-to-MESOSCOPE/MESOSCOPE-to-Falkor. All six percentile intervals nevertheless crossed zero, so this sensitivity supports a positive point effect but not a bootstrap sign claim after source refitting and RT-block resampling.

The RT-block intervals propagate source refitting and local retention-time dependence. The deterministic file deletion repeats per-file aggregation and model fitting. Neither analysis can recover unavailable compound/adduct identifiers or represent additional laboratories.
