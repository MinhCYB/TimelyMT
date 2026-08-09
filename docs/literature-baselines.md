# Literature Baselines

These components are paper-faithful semantic adaptations where the core
decision or metric can be mapped to TimelyMT. They are not exact reproductions
of the papers' original systems, architectures, audio front ends, language
pairs, or training regimes.

## Local Agreement LA-2 Adaptation

- Method name: `local_agreement_la2`.
- Source: Local Agreement decoding as used for incremental speech translation,
  in particular Liu et al. (2020), *Low-Latency Sequence-to-Sequence Speech
  Recognition and Translation by Partial Hypothesis Selection*.
- Original high-level idea: compare recent translation hypotheses and emit the
  longest target prefix on which they agree; already emitted output is final.
- TimelyMT adaptation: after four causal lexical source tokens, translate every
  growing source prefix within a consecutive source unit capped at TimelyMT's
  frozen 48-token maximum. Tokenize normalized EnViT5 output
  with exact `translated_text.split()`. Compare the current and immediately
  previous hypotheses (LA-2), emit only target tokens after the immutable
  emitted-prefix length and through their longest common token prefix, and
  never emit those tokens again. At talk termination, append the current
  hypothesis suffix beginning at the immutable emitted-token count. Maximum
  unit termination uses the same flush and starts a fresh LA-2 unit.
- Runtime information: arrived source tokens, current/previous normalized
  system hypotheses, and already emitted token count. No future source or
  reference is available.
- Training oracle information: none; this is a heuristic baseline.
- Meaningfulness: it preserves Local Agreement's defining common-prefix and
  irrevocable-output semantics while using the same frozen translator and
  source stream as every TimelyMT strategy.
- Limitations: source updates are lexical tokens rather than audio/ASR chunks;
  cumulative text translation replaces the original system's speech pipeline;
  final flush is a deterministic TimelyMT termination rule. Commit statistics
  count stable-prefix output events. Their source-unit size and duration cover
  source tokens newly observed since the preceding output event, while target
  latency uses the event's actual observation position.

`local_agreement_style_k2` and `local_agreement_style_k3` remain separate
TimelyMT-specific source-boundary heuristics. They compute an LCP ratio and,
when it reaches 0.90, commit the current whole unit hypothesis. They are not
renamed or presented as paper-faithful Local Agreement.

## Zhang-2020 Meaningful Unit Adaptation

- Method name: `mu_zhang2020`.
- Source: Ruiqing Zhang, Chuanqiang Zhang, Zhongjun He, Hua Wu, and Haifeng
  Wang (2020), *Learning Adaptive Segmentation Policy for Simultaneous
  Translation*, EMNLP 2020, DOI `10.18653/v1/2020.emnlp-main.178`.
- Original high-level idea: learn source segmentation jointly informed by
  possible translations so segments form Meaningful Units compatible with
  translation.
- TimelyMT training target: for candidate remaining-source prefix `S_t`, obtain
  `H_t = EnViT5(S_t)` and the training-only oracle `H_full` from the same start
  through the full admissible source unit: the earlier of 48 lexical source
  tokens or talk termination. The candidate is a meaningful-unit
  boundary when `LCP(H_t, H_full) / max(1, len(H_t.split())) >= 0.90`. Maximum
  length and talk end remain forced boundaries. This supervision is generated
  independently under `data/policy/mu_zhang2020`; TimelyMT future-stability
  pseudo-labels are neither reused nor renamed.
- Model: balanced logistic regression. Text features are TF-IDF word unigrams
  and bigrams (`min_df=1`, at most 10,000 per field) over only
  `current_source_text` and `current_hypothesis_text`. Numeric features are
  current source token count, source character count, elapsed simulated source
  clock, current target token count, and source/target length ratio.
- Runtime information: only the current uncommitted source buffer, its current
  frozen-translator hypothesis, and the five local numeric values. The
  classifier runs sequentially at threshold 0.50. It receives no committed
  target/source history, P2 history, reference, alignment, future translation,
  or gold target.
- Training oracle information: full admissible remaining-unit frozen-translator
  hypothesis and derived prefix-preservation ratio, in TRAIN/DEV supervision
  only. TEST supervision is rejected.
- Meaningfulness: the policy retains the paper's translation-aware learned
  segmentation principle and is compared with the same translator and dataset.
- Limitations: this is not the paper's neural segmentation architecture,
  reinforcement objective, NMT training, ASR setup, or language pair. Prefix
  compatibility with a full admissible-unit translation is the closest clean
  oracle available through TimelyMT's frozen translator interface.

## LAAL

- Metric: Length-Adaptive Average Lagging from Papi et al. (2022),
  *Over-Generation Cannot Be Rewarded: Length-Adaptive Average Lagging for
  Simultaneous Speech Translation*, DOI `10.18653/v1/2022.autosimtrans-1.2`.
- For source length `|X|`, hypothesis length `|Y*|`, reference length `|Y|`,
  and the existing emitted-target source-consumption sequence `g(t)`, define
  `gamma = max(|Y|, |Y*|) / |X|` and
  `tau = min { t : g(t) >= |X| }`, or `|Y*|` when source completion is absent.
  Then `LAAL = (1/tau) sum_{t=1..tau} [g(t) - (t-1)/gamma]`.
- The evaluator computes talk-level LAAL and aggregates it weighted by emitted
  hypothesis token count, matching TimelyMT's existing AL aggregation. Standard
  AL remains unchanged and is reported separately.
- Reference target length is loaded and passed only inside `evaluate`; rollout,
  strategies, policies, prediction records, and translator requests do not
  receive it.
