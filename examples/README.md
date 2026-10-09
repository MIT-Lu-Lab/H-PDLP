# Example Instance

`2club200v15p5scn.mps.gz` is an unmodified MIPLIB 2017 instance, submitted by Balabhaskar Balasundaram. It models a maximum 2-club problem and contains 200 binary variables, 17,013 constraints, and 104,811 nonzeros.

- [MIPLIB instance page](https://miplib.zib.de/instance_details_2club200v15p5scn.html)
- [Original compressed MPS file](https://miplib.zib.de/WebData/instances/2club200v15p5scn.mps.gz)

This is third-party benchmark data, separate from the H-PDLP solver source. See the linked MIPLIB page for the instance's provenance and literature reference.

From the repository root, after [building H-PDLP](../README.md#installation):

```bash
mkdir -p result
./build/hpdlp examples/2club200v15p5scn.mps.gz result/
```

H-PDLP treats the integer variables as continuous and solves the **LP relaxation**. At default tolerances, expect `OPTIMAL` and a primal objective of approximately **-121.2**; the last digits depend on the stopping tolerances. The integer objective of -70 on the MIPLIB page does not apply to this relaxation.

Check `result/2club200v15p5scn_summary.txt` for the termination status, primal/dual objective values, and residuals. This example is an installation check, not a performance benchmark. See the [CLI Guide](../docs/CLI.md#output-files) for the output file format.
