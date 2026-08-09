# Architecture

TimelyMT models the timing decision in incremental English-to-Vietnamese cabin translation.

```text
Continuous bilingual talks -> streaming dataset builder -> source prefixes
-> frozen EN-VI translator -> translation hypotheses -> commit strategy
-> streaming evaluation
```

Commit strategies include Fixed-N, Fixed-Time, LocalAgreement, and a future learned LISTEN/COMMIT policy. The policy may use current and previous hypotheses, source and target history, technical context, and numeric stability features.

`data/streaming/` is the central data branch. `data/policy/` stores derived prefixes, hypotheses, and pseudo-labels. The translator is a frozen abstraction under `src/timelymt/translator/`; translator training is outside the active architecture.
