# Policy V2 TEST Plan

Policy V2 remains a **POST-HOC EXPLORATORY DEV EXTENSION**. Held-out TEST only measures whether the frozen DEV observation generalizes.

- Primary: `v2_P2_0.50`
- History ablation: `v2_P0_0.50`, `v2_P1_0.50`, `v2_P2_0.50`
- References: `learned_P1_0.60`, `local_agreement_la2`, `fixed_n_8`, `fixed_time_3200`
- Pipeline: frozen prediction, frozen evaluation, reporting
- Forbidden: training, fine-tuning, pseudo-label generation, search, selection, re-selection, configuration changes, and checkpoint replacement

No TEST winner is calculated. The primary identity was selected on DEV before TEST access.
