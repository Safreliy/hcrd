# Hierarchical Convexity-Run Decomposition (HCRD)

Reference implementation and reproducibility materials for *Hierarchical
Convexity-Run Decomposition for Interpretable LC--MS Peak-Quality Curation:
Finite Chord-Lobe Theory and Cross-Study Validation*.

**Saveliy Baturin**  
Independent Researcher

![HCRD algorithm overview](paper/figures/method_overview.png)

HCRD partitions a sampled graph into maximal runs of one discrete-curvature
sign, replaces each run by its endpoint chord, stores the signed residual lobe,
and repeats on the retained knots. The hierarchy is finite, nested, exactly
reconstructing, and interpretable at every level.

## Features

- dense and sparse multilevel decompositions with exact reconstruction;
- support for uniformly and irregularly sampled signals;
- signed lobe support, amplitude, shape, area, and quadratic-energy summaries;
- deterministic process-parallel decomposition of independent signals;
- stable proximal and persistence-based companion representations;
- calibrated finite-dictionary and continuous-family lobe scans;
- tests, experiment protocols, compact results, and source-data manifests.

## LC--MS application

The principal real-data application is quality curation of pre-bounded LC--MS
extracted-ion chromatograms. A single logistic pipeline was transferred between
two independently labelled HILIC studies without target refitting.

| Transfer | qscore AP | HCRD-1+Q AP | HCRD-8+Q AP | HCRD-8 - qscore (95% CI) |
|---|---:|---:|---:|---:|
| Falkor to MESOSCOPE | 0.7774 | 0.8516 | **0.8955** | +0.1181 [0.0613, 0.1781] |
| MESOSCOPE to Falkor | 0.7984 | 0.8446 | **0.8990** | +0.1005 [0.0559, 0.1470] |

The score is an independent global-window reimplementation of the published
two-component definition. The Holm-adjusted p-value was 0.00040 in both
directions. HCRD-8 had positive AP differences over HCRD-1 in both transfers,
with multiplicity-adjusted support in one. An equal-dimensional 2847-variable
Gaussian-derivative control tied HCRD-8 in Falkor-to-MESOSCOPE; HCRD-8 exceeded
it by 0.1175 AP in the reverse transfer (95% CI [0.0624, 0.1654]). On the
conditionally labelled Pttime subset, HCRD-8+Q achieved residual-error AP
0.5085, compared
with 0.0391 for qscore and 0.2845 for HCRD-1+Q. At 1%, 5%, and 10% review
budgets it retrieved 4, 7, and 8 of 17 bad cases, versus 0, 0, and 1 for
qscore. On the independent TARDIS/FAME
shift, HCRD-8+Q improved AP by 0.0725 in point estimate, while the confidence
interval crossed zero and a compact conventional feature bank was point-best.
These results identify pre-bounded compact-lobe curation as the supported
application setting and show that hierarchy depth can depend on acquisition
conditions.

![Independent LC-MS evidence](paper/figures/lcms_evidence.png)

## Installation

Python 3.11 or later is recommended.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python examples/quickstart.py
```

Optional dependency groups are available for LC--MS, PPG, anomaly-detection,
classification, RUL, and comparator experiments; see `pyproject.toml`.

## Quick start

```python
import numpy as np
from hcrd import decompose, decompose_sparse, level_energies

x = np.linspace(0.0, 1.0, 256)
y = np.sin(2 * np.pi * x) + 0.25 * np.exp(-((x - 0.6) / 0.08) ** 2)

dense = decompose(y, x)
np.testing.assert_allclose(dense.reconstruct(), y, atol=1e-12)

sparse = decompose_sparse(y, x)
print(sparse.knot_sets)
print(level_energies(sparse))
```

Independent signals can be processed in deterministic order:

```python
from hcrd import decompose_sparse_batch

hierarchies = decompose_sparse_batch(signals, backend="process", workers=4)
```

## Reproducing the main LC--MS experiment

```bash
python -m pip install -e ".[dev,lcms]"
python experiments/download_ms_metrics_e2.py
python experiments/run_ms_metrics_e2.py extract-dataset --help
python experiments/run_ms_metrics_e2.py fit-evaluate --help
python experiments/run_ms_metrics_e2_matched_capacity.py --help
```

The downloader verifies the official archives and pinned source revision. Raw
mzML files and multi-gigabyte intermediate arrays are rebuilt locally and are
not redistributed. Frozen labels, compact model outputs, bootstrap statistics,
and hashes are included under `data/manifests/`, `results/ms_metrics_e2/`, and
`reports/ms_metrics_e2.md`.

Commands for all other experiments are listed in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Repository structure

| Path | Contents |
|---|---|
| `src/hcrd/` | Transform, sparse hierarchy, feature extraction, and scans |
| `tests/` | Unit, regression, and mathematical-property tests |
| `experiments/` | Data preparation and experiment entry points |
| `docs/` | Dataset-specific protocols and methodological notes |
| `theory/` | Proof structures and mathematical supplements |
| `results/` | Compact numerical outputs |
| `reports/` | Experiment summaries |
| `paper/` | LaTeX source, figures, and compiled preprint |
| `data/manifests/` | Source URLs, checksums, and dataset partitions |

The integrity manifest can be checked with:

```bash
python scripts/verify_release.py
```

## Paper and citation

The compiled manuscript is [`paper/hcrd_preprint.pdf`](paper/hcrd_preprint.pdf).
Citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## Data and licenses

Third-party datasets and comparator repositories are not redistributed.
Download scripts, source URLs, archive hashes, protocols, and compact outputs
are included where licensing permits.

Code is licensed under the MIT License (`LICENSE`). The manuscript,
documentation, protocols, original figures, and original result files are
licensed under CC BY 4.0 (`LICENSE-CONTENT.md`). Third-party materials retain
their original terms.
