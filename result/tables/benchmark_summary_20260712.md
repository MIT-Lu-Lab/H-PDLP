# Benchmark summary: cuHSDLP vs cuPDLPx (20260712, tol 1e-4)

| Suite | n | solved HSD | solved PDLPx | SGM time HSD (s) | SGM time PDLPx (s) | time ratio | SGM iters HSD | SGM iters PDLPx | iter ratio |
|---|---|---|---|---|---|---|---|---|---|
| MIPLIB | 383 | 380 | 380 | 2.0 | 2.1 | 0.93x | 6238 | 6916 | 0.90x |
| Mittelmann | 49 | 48 | 48 | 14.8 | 9.7 | 1.52x | 34911 | 26348 | 1.33x |

SGM population per suite: MIPLIB 381/383 (cap 18000s), Mittelmann 49/49 (cap 15000s)

## Termination disagreements

| Suite | Instance | cuHSDLP | time (s) | iters | cuPDLPx | time (s) | iters |
|---|---|---|---|---|---|---|---|
| Mittelmann | Dual2_5000 | TIME_LIMIT | 15000.3 | 4001000 | OPTIMAL | 14.5 | 5200 |
| Mittelmann | ns1687037 | OPTIMAL | 61.7 | 1066800 | TIME_LIMIT | 15000.0 | 313672000 |
