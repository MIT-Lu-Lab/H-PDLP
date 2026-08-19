# Plain HSD-PDHG trajectory summaries

All linearity windows span 5,000 iterations, slide by one 200-iteration
trace sample, and require $R^2 \geq 0.99$. For $\|x\|_2$ and $\|y\|_2$,
the sampled relative second-difference filter is $q \leq 1\%$.

- `shape_only_*`: no minimum change-amplitude requirement.
- `growth_ge_1pct_*`: absolute fitted endpoint change divided by the
  window mean is at least 1%; rising and falling are classified separately.
- `all_phase_duration_*`: duration statistics over all projection-signature
  phases; these do not condition on linearity windows.
- `shape_only_linear_iteration_phase_duration_*`: each phase intersecting at
  least one shape-only linear window is counted once.
- `growth_ge_1pct_linear_iteration_phase_duration_*`: each phase intersecting
  at least one linear window with at least 1% fitted change is counted once.

The superseded prefix-sum $R^2$ outputs are intentionally excluded because
their local variance calculation was numerically unstable on long traces.
