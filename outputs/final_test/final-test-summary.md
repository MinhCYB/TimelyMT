# TimelyMT Final Held-Out TEST Evaluation

## Protocol

TEST was opened only after the design and artifact freeze. No post-TEST tuning is permitted. No prepared TEST context was constructed; P3 used only the exact float32 zero vector of dimension 384.

The causal prepared-context REAL/ZERO result remains a DEV controlled experiment. This TEST phase does not claim to independently validate prepared-context benefit on unseen context-bearing talks.

## Evaluated Configurations

- V1: `learned_P1_0.60`.
- V2: `v2_P2_0.50`.
- P3: no pre-existing selected threshold was found, so the pre-registered frozen grid `0.30, 0.40, 0.50, 0.60, 0.70` was evaluated with ZERO context. No TEST-based selection is made.

## TEST Results

| model/configuration | threshold | BLEU | chrF2 | AL | LAAL | commits |
|---|---:|---:|---:|---:|---:|---:|
| learned_P1_0.60 | 0.60 | 23.3764 | 56.2944 | 1.2208 | 41.2577 | 329 |
| v2_P2_0.50 | 0.50 | 21.0542 | 55.3374 | -0.0696 | 40.1514 | 465 |
| p3_global_zeroctx_0.30 | 0.30 | 19.2575 | 55.3805 | 10.5463 | 13.5856 | 624 |
| p3_global_zeroctx_0.40 | 0.40 | 19.2526 | 55.4658 | 7.3587 | 16.9839 | 603 |
| p3_global_zeroctx_0.50 | 0.50 | 21.9787 | 56.7129 | 13.6879 | 20.3813 | 514 |
| p3_global_zeroctx_0.60 | 0.60 | 20.5661 | 55.1103 | 18.0144 | 55.3536 | 453 |
| p3_global_zeroctx_0.70 | 0.70 | 22.7885 | 55.6135 | 25.0923 | 56.6111 | 306 |

## DEV vs TEST Descriptive Comparison

These are descriptive frozen-split comparisons only. They do not define a new operating point.

| configuration | DEV BLEU | TEST BLEU | DEV chrF2 | TEST chrF2 | DEV AL | TEST AL | DEV LAAL | TEST LAAL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| learned_P1_0.60 | 27.0862 | 23.3764 | 61.3961 | 56.2944 | -8.9187 | 1.2208 | 61.2543 | 41.2577 |
| v2_P2_0.50 | 25.9932 | 21.0542 | 61.4553 | 55.3374 | -19.4171 | -0.0696 | 19.5891 | 40.1514 |
| p3_global_zeroctx_0.30 | 25.2457 | 19.2575 | 61.9753 | 55.3805 | 1.0477 | 10.5463 | 4.9197 | 13.5856 |
| p3_global_zeroctx_0.40 | 25.0922 | 19.2526 | 61.7692 | 55.4658 | -1.8004 | 7.3587 | 7.5924 | 16.9839 |
| p3_global_zeroctx_0.50 | 25.8650 | 21.9787 | 61.8864 | 56.7129 | 7.0279 | 13.6879 | 25.6110 | 20.3813 |
| p3_global_zeroctx_0.60 | 25.8900 | 20.5661 | 61.1519 | 55.1103 | 0.3896 | 18.0144 | 51.1500 | 55.3536 |
| p3_global_zeroctx_0.70 | 28.7583 | 22.7885 | 62.0540 | 55.6135 | 23.0324 | 25.0923 | 90.0666 | 56.6111 |

## Interpretation

The table reports held-out behavior as observed. Relative differences and DEV-to-TEST shifts are descriptive only. No P3 threshold is designated best or selected from TEST. Because every P3 TEST run used ZERO context, these results neither support nor refute prepared-context benefit on TEST.

## Post-TEST Freeze

No model, threshold, feature, training procedure, prepared-context representation, or metric will be changed based on these TEST results.
