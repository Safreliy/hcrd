# Runtime benchmarks and reproducibility checks

This document records deterministic reruns, controlled runtime measurements,
and SHA-256 hashes for selected experiment artifacts.

## Scientific-result recalculation

The S2 quadratic-guide confirmation was rerun into
`results/stable_confirmation_s2_rerun`.  All six outputs were byte-identical to
the originals:

| File | SHA-256 |
|---|---|
| `aggregate.json` | `D20B50EDB341D97A600E38F0678C2496ECAFFBEF2417024584B722F425A5D7DE` |
| `comparisons.json` | `405260CF50CAEF7AD7F4462F3382CE29E4FC7CBC0F7FC3131BCA6E9DCE32F146` |
| `latent_summary.csv` | `17EF728973919B27B32FF560046FB3412CEF455217E5A3176F46E4BEEC762BDB` |
| `protocol.json` | `37316266D5B7A29BC9C42AE9DEB816EE6BA74A93A068D3B58E4C5105DB4DCE6F` |
| `stability.csv` | `A16C942BDDFCE7678DA444DA2911230AFC0679A4BA2093A335EFDFB22BBB49EE` |
| `trials.csv` | `28F62F1EEBFDB327ADA5621BEAAB22D56BA59EFFBC1BC251440FB351BAEAEA13` |

The locked QTDB R2 confirmation was rerun into
`results/qtdb_confirmation_r2_rerun`.  All five outputs were byte-identical:

| File | SHA-256 |
|---|---|
| `aggregate.json` | `E718BF8DB395ADDBA51CE29FDB90AC556FEB3CF1570F0545AA6582A5526A63D9` |
| `comparisons.json` | `965CB741C3165D1BDAAEBFC04B2772A8197F46A2BBEB3B598C18714DDB30396F` |
| `record_summary.csv` | `76684BC86564D95DBFE7378AE91CFB0211C88AE97A298EAEC803CB101D5F44EB` |
| `run_metadata.json` | `144314BC8A6289E3EB1592EA2298FB753CE14DF085AFF4A315D2C3D9674ED922` |
| `trials.csv` | `BF3FD800991BF1FA8376E31CD2CDDEB63FC5B5702C8C81E4FA357434696F55DA` |

The rerun artifacts are byte-identical to the reported outputs.

## Controlled P1 runtime result

The protocol in `docs/parallel_runtime_protocol.md` was frozen before execution.
The workload was the same 384 CWRU windows used in R1, with five repetitions in
randomized mode order.  The benchmark began after a five-sample whole-system
load average below 20%, fixed compiled-library worker counts to one, and
included process-pool startup and serialization in every timed batch.

| Backend | Workers | Median (s) | IQR (s) | Windows/s | Speedup |
|---|---:|---:|---:|---:|---:|
| serial | 1 | 10.279 | 9.520--12.041 | 37.36 | 1.00x |
| thread | 4 | 18.533 | 17.083--23.848 | 20.72 | 0.55x |
| thread | 8 | 17.249 | 16.660--17.639 | 22.26 | 0.60x |
| process | 2 | 5.843 | 5.805--6.311 | 65.72 | 1.76x |
| process | 4 | 4.767 | 4.387--6.102 | 80.55 | 2.16x |
| process | 8 | 4.163 | 3.551--5.533 | 92.24 | 2.47x |
| process | 16 | 5.135 | 4.149--5.813 | 74.79 | 2.00x |

All 35 timed matrices and the new serial reference had SHA-256
`ef6621468e1a5519109c294e146eadc0f84ed84c1b0da0b72db1abd815751352`
and were bitwise identical to the earlier R1 matrix.  Eight processes were the
best tested mode on this Intel Core Ultra 7 155H (16 physical, 22 logical CPUs),
but this optimum is hardware- and batch-size-specific.  Threads regress because
the current knot walk spends substantial time in Python code under the GIL.

The valid parallel claim is therefore batch/channel/record parallelism across
independent signals.  Levels within one signal remain sequential because every
new eligible knot set depends on the preceding level.

## Sparse implementation and load-gated recalculation

The knot-only implementation traverses and stores only the nested knot sets;
dense baseline/detail arrays are materialized only on request.  On all 384 P1
windows its dense materialization reproduced the saved R1 feature matrix
bitwise with SHA-256
`ef6621468e1a5519109c294e146eadc0f84ed84c1b0da0b72db1abd815751352`.

Protocol P2R required five consecutive one-second CPU samples, **each** no
greater than 20%, before every timed trial. All 25 trials reproduced knot digest
`2aadbfdae86a2abf34b5821d0fd0e54884157fba46b6a23b7cd89edba159b62f`.

| Representation/backend | Workers | Median (s) | IQR (s) | Windows/s | Speedup vs sparse serial |
|---|---:|---:|---:|---:|---:|
| dense serial | 1 | 14.461 | 10.140--14.478 | 26.55 | 0.08x |
| sparse serial | 1 | 1.217 | 1.209--1.499 | 315.63 | 1.00x |
| sparse process | 2 | 1.248 | 1.162--1.315 | 307.61 | 0.97x |
| sparse process | 4 | 0.976 | 0.896--1.080 | 393.63 | 1.25x |
| sparse process | 8 | 1.421 | 1.131--1.782 | 270.33 | 0.86x |

Thus sparse storage was 11.89x faster than dense materialization on this fixed
batch.  Four processes gave a further 1.25x, while eight did not amortize pool
and serialization overhead.

P3 froze a tenfold larger 3840-window batch and used the same strict per-trial
gate.  All 12 outputs reproduced knot digest
`68e1a16d3c9fa2861111b8e8aecfb0c6c23cb5c8bac9a5e31deb4d373c079733`.

| Backend | Workers | Median (s) | IQR (s) | Windows/s | Speedup |
|---|---:|---:|---:|---:|---:|
| serial | 1 | 13.682 | 12.208--15.269 | 280.65 | 1.00x |
| process | 2 | 11.911 | 10.338--11.949 | 322.40 | 1.15x |
| process | 4 | 8.669 | 7.650--9.477 | 442.94 | 1.58x |
| process | 8 | 6.963 | 6.816--7.399 | 551.45 | 1.96x |

This supports exact outer parallelism for sufficiently large batches, not an
ideal linear-scaling claim.  Python threads remain inappropriate for the
GIL-bound knot walk; processes are the CPU-parallel backend.

| Study | Artifact | SHA-256 |
|---|---|---|
| P2 diagnostic | `summary.json` | `03051CE5D437448055A404CC44CC043CCCF29DECD72E311940068BABA72D004D` |
| P2 diagnostic | `trials.csv` | `E19A8B307C28F771CABF7A6EEE4B5F3F1BD770C5CBF1855F8CB610F585CDCBEB` |
| P2R replacement | `summary.json` | `A6BFA86B03D8768A8ABB5BA8D52A12414025091206135C28F0C414D766F91EA7` |
| P2R replacement | `trials.csv` | `D82850BF4A57EAEE0334C409B15568B0C160D72C66E2F3899E38ABEE4FBE1064` |
| P3 throughput | `summary.json` | `BC6E3148224DEA331D95678BC4135AFE64B991EC622461CF38F8A2A9BDBCA00B` |
| P3 throughput | `trials.csv` | `6A27BC5C6E6073B12433426FEC3F58D0263176DDA04C65180ED7D16D51D2611B` |

## External-comparison artifact hashes

Key SHA-256 digests allow regenerated artifacts to be compared without relying
on timestamps.

| Study | Artifact | SHA-256 |
|---|---|---|
| E2 CEEMDAN | protocol | `0A212E07C9EA2DF6CB69DD378E4B4F7309544A7CCD14DC96C8802168DD404846` |
| E2 CEEMDAN | trials | `C20BAC0EDB5ED4097F436E1452F420269B02822FAC80A994383583BA287717BB` |
| E2 CEEMDAN | comparisons | `FBF8BADDE9FF8ADC5DD8D1C93B142F707A93EE5A0AD140E7B81BD5F6F9335824` |
| E3 Iterative Filtering | protocol | `8F174FB13B81CD65E16F5AB82B73ED0EF613EC69FCBA8A405D634EDCE26C2314` |
| E3 Iterative Filtering | trials | `0FDB04C43C53D4A00F5242B9A3E55B9A72E2C06F1B99CD18799B6118ED9ED6F8` |
| E3 Iterative Filtering | comparisons | `2707D05029BDBDB22EF5C52D4D9C0F2A5A638A58450B5EF4EEE1EE21315DAC33` |
| R3 NeuroKit | protocol | `BD0AA6BD3EB76838E931725F1D3AAC73DF0554111C003C360B8925804C576D66` |
| R3 NeuroKit | trials | `D115C4CF5B8D85E9E92C09A3A6E60E41A03DD24327188EFDF0A03B403FB31D5E` |
| R3 NeuroKit | comparisons | `BD88F19E14A6D19994B926869AA04BD0B4C71146ED27D86C0FF0519BFAEFE21E` |
