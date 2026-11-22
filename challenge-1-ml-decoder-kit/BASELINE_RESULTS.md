# Step 1: baseline results

`python baseline.py --p <p> --shots 10000 --workers 8`, seed 1, 10,000 shots per basis.
Machine: Apple M5 (4 performance + 6 efficiency cores), Python 3.12.13, ldpc 2.4.1, relay-bp 0.2.2.
Logs: `baseline_p0.003.log`, `baseline_p0.004.log`, `baseline_p0.005.log`.

| p | decoder | failures X, Z | rate per cycle | authors' failures X, Z (shots) | authors' rate | mean time | p99 | run time, 8 workers |
|---|---------|---------------|----------------|--------------------------------|---------------|-----------|-----|---------------------|
| 0.003 | BP-OSD | 36, 23 | 4.9e-4 +- 0.6e-4 | 282, 316 (100,000) | 5.0e-4 | 0.11 s | 0.72 s | 4.4 min |
| 0.003 | relay-BP | 3, 4 | 5.8e-5 +- 2.2e-5 | 32, 40 (100,000) | 6.0e-5 | 0.09 s | 0.59 s | |
| 0.004 | BP-OSD | 266, 267 | 4.5e-3 +- 0.2e-3 | 244, 265 (10,000) | 4.3e-3 | 0.27 s | 0.76 s | 12.0 min |
| 0.004 | relay-BP | 52, 40 | 7.7e-4 +- 0.8e-4 | 53, 44 (10,000) | 8.1e-4 | 0.29 s | 4.9 s | |
| 0.005 | BP-OSD | 1310, 1344 | 2.36e-2 +- 0.05e-2 | 660, 651 (5,000) | 2.3e-2 | 0.50 s | 0.92 s | 41.8 min |
| 0.005 | relay-BP | 418, 425 | 7.2e-3 +- 0.3e-3 | 216, 205 (5,000) | 7.1e-3 | 1.45 s | 8.1 s | |

- Mean time and p99 cover one X shot plus one Z shot.
- Run time is the wall clock for both decoders at that p. The README's 8-core run times are 6, 17 and 58 min.
- 4 of the 8 workers run on efficiency cores, which raises the per-shot times. With 5 workers at p = 0.004
  the mean times were 0.17 s (BP-OSD) and 0.23 s (relay-BP), with the same failure counts.
