# E1 LasHeR test preliminary result

This is the controller output after both complete 245-sequence test shards; a fresh three-benchmark audit is pending.

| version | PR | NPR | SR | mean ms | FPS |
|---|---:|---:|---:|---:|---:|
| user baseline | 72.740000 | 69.160000 | 57.840000 | — | — |
| E0 full C882 | 74.749703 | 71.067672 | 59.089007 | 113.817 | 8.786 |
| E1 full C882 | 67.612357 | 62.545797 | 50.842354 | 138.345 | 7.228 |

E1 is below the user baseline by −5.127643 / −6.614203 / −6.997645 percentage points (PR/NPR/SR), and below E0 by −7.137346 / −8.521875 / −8.246652 points. Per-sequence E1 versus the historical C20 report has 61/245 PR wins, 57/245 NPR wins and 52/245 SR wins; this comparison is descriptive because the E0 artifact lineage differs.

The route statistics expose a concrete failure chain. E0 used TIR on 23,947/220,458 causal calls (10.86%) and made 185,027 auxiliary calls (83.93%). E1 used TIR on 114,975/220,458 calls (52.16%) and made only 14,826 auxiliary calls (6.72%). This shifts both the primary modality and STOP/query behavior. The evidence supports a routing/gain calibration problem; it does not isolate which training head caused it.

Keep the current RGBT210 and RGBT234 evaluation unchanged. After all three benchmarks close, audit raw coverage and score arithmetic, then run a fixed-weight diagnostic comparing the E1 quality/gain heads and route decisions against the protected E0 path before deciding any matched-budget repair.
