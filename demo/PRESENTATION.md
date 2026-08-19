# TimelyMT Demo Presentation Script

This 2-3 minute walkthrough uses the static DEV artifact replay. Start at the default `first-divergence` bookmark: event 131, `00:53.000`.

## 1. Problem - 15 seconds

"TimelyMT must decide when enough source has arrived to commit an irreversible translation. The question here is not whether one translation is better. It is whether prepared context changes the policy's commitment timing."

## 2. Controlled Experiment - 20 seconds

"Both sides replay the same observed English source with the same P3 checkpoint, the same EnViT5 source-only translator design, and the same 0.60 threshold. Prepared source exists in both conditions. The only intervention is that ZERO sets the prepared-context policy embedding to zero."

Point to the Prepared Knowledge panel: REAL effective norm `1.0`; ZERO effective norm `0.0`.

## 3. Streaming Replay - 20 seconds

"Each incoming source token is timestamped in the trace. The viewer advances along that same source observation clock. Between commits, each policy accumulates its own uncommitted source buffer."

"With fewer than four buffered tokens, the system waits without running the translator or policy. Once inference is allowed, LISTEN keeps the source buffered while COMMIT freezes the current translation and starts a new source segment."

"After REAL and ZERO make different commit decisions, they continue hearing the same incoming source token, but their accumulated buffers can differ. Different later candidate translations are then downstream consequences of those different source spans."

## 4. First Divergence - 30 seconds

"At event 131, 00:53, both conditions have the same candidate source span and the same Vietnamese candidate translation. REAL has p(COMMIT) `0.631034`, which crosses the `0.60` threshold, so it commits. ZERO has `0.575424`, below the same threshold, so it listens. The probability gap is `+0.055609`."

"This is causal policy sensitivity, not quality superiority: the candidate span, translation, checkpoint, and threshold are held constant. Only the prepared-context policy input differs."

## 5. Cascade - 30-45 seconds

Select **Downstream commit-boundary cascade** from the bookmark selector.

"After an earlier timing difference, the policies can reach a later observation with different candidate source boundaries. This panel now marks that direct candidate comparison is no longer valid: later translations may differ indirectly because the source spans differ. The committed-output rows make the irreversible boundary history visible."

## 6. Aggregate Finding - 20 seconds

"For this Sims DEV talk at threshold 0.60, the descriptive summary is `+0.95 BLEU`, `+0.63 chrF2`, and `318` versus `299` commits. This is one context-bearing DEV talk and is descriptive only, not evidence of general improvement."

## 7. Conclusion - 15 seconds

"Prepared context affects commit timing in this controlled replay. However, prepared-global-v0 is not yet a robust general improvement. The appropriate next step is to preserve this result, rehearse the demonstration, and avoid changing research or model code for presentation."
