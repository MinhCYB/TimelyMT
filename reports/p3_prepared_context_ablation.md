# Controlled P3 Prepared-Context Ablation

## Design

Comparison is **P3_REAL - P3_ZERO_CONTEXT** using the same P3_GLOBAL checkpoint. It tests inference-time reliance on the prepared-global input by the trained policy; it does not establish that another training design could not use context differently.

## Aggregate DEV

| threshold | dBLEU | dchrF2 | dAL | dLAAL | d commits |
|---:|---:|---:|---:|---:|---:|
| 0.3 | -0.03 | -0.02 | -2.29 | -2.29 | 2 |
| 0.4 | -0.01 | -0.07 | 2.23 | 2.22 | 8 |
| 0.5 | -0.22 | 0.04 | 0.51 | -2.07 | 18 |
| 0.6 | 0.21 | 0.14 | 0.38 | -2.6 | 19 |
| 0.7 | -0.44 | -0.08 | -3.88 | -9.82 | 44 |

## Per-Talk and Commit Behavior

| threshold | context | talk | dBLEU | dchrF2 | real/zero commits | real/zero mean span | empty artifacts identical |
|---:|---|---|---:|---:|---:|---:|---|
| 0.3 | empty | ted-jeff-dean-ai-smart | 0 | 0 | 831/831 | 4.09/4.09 | True |
| 0.3 | empty | ted-luis-von-ahn-crowdsourcing | 0 | 0 | 790/790 | 4.12/4.12 | True |
| 0.3 | bearing | ted-sims-witherspoon-ai-climate | -0.15 | -0.07 | 412/410 | 4.1/4.12 | None |
| 0.4 | empty | ted-jeff-dean-ai-smart | 0 | 0 | 822/822 | 4.13/4.13 | True |
| 0.4 | empty | ted-luis-von-ahn-crowdsourcing | 0 | 0 | 780/780 | 4.18/4.18 | True |
| 0.4 | bearing | ted-sims-witherspoon-ai-climate | -0.04 | -0.29 | 395/387 | 4.28/4.36 | None |
| 0.5 | empty | ted-jeff-dean-ai-smart | 0 | 0 | 787/787 | 4.32/4.32 | True |
| 0.5 | empty | ted-luis-von-ahn-crowdsourcing | 0 | 0 | 713/713 | 4.57/4.57 | True |
| 0.5 | bearing | ted-sims-witherspoon-ai-climate | -0.93 | 0.17 | 359/341 | 4.7/4.95 | None |
| 0.6 | empty | ted-jeff-dean-ai-smart | 0 | 0 | 646/646 | 5.26/5.26 | True |
| 0.6 | empty | ted-luis-von-ahn-crowdsourcing | 0 | 0 | 578/578 | 5.64/5.64 | True |
| 0.6 | bearing | ted-sims-witherspoon-ai-climate | 0.95 | 0.63 | 318/299 | 5.31/5.65 | None |
| 0.7 | empty | ted-jeff-dean-ai-smart | 0 | 0 | 435/435 | 7.81/7.81 | True |
| 0.7 | empty | ted-luis-von-ahn-crowdsourcing | 0 | 0 | 344/344 | 9.47/9.47 | True |
| 0.7 | bearing | ted-sims-witherspoon-ai-climate | -1.99 | -0.33 | 226/182 | 7.47/9.28 | None |

## Interpretation

Positive quality deltas favor real context; negative AL/LAAL deltas favor real context on latency. For naturally empty-context talks, the artifact equality column must be `True` under deterministic inference. A `False` value is an implementation/runtime anomaly to investigate before interpreting context-bearing effects.
