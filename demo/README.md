# TimelyMT Static Research Demo

This is a static, artifact-replay viewer for the controlled DEV comparison of REAL_CONTEXT and ZERO_CONTEXT for `ted-sims-witherspoon-ai-climate` at threshold `0.60`.

## Launch

From the repository root in Windows PowerShell:

```powershell
python -m http.server 8000
```

Open [http://localhost:8000/demo/](http://localhost:8000/demo/).

The viewer is deliberately static: it requires no model, runtime, Node/npm package, backend API, inference, rollout, training, or evaluation. It replays only the validated DEV artifacts.

## Required Artifacts

- `outputs/demo_traces/sims-real-0.60.json`
- `outputs/demo_traces/sims-zero-0.60.json`
- `outputs/demo_traces/sims-0.60-bookmarks.json`

At startup the browser validates the trace-pair identity, `0.60` threshold, checkpoint SHA, source-token count, event count, mode pair, and every synchronized `event_index`, `source_token_end`, and `observation_ms`. It also rejects bookmarks that do not reference a synchronized event. A validation failure is blocking and visible.

## Use

The recommended start is the `first-divergence` bookmark, event 131 at `00:53.000`. Use the step/playback controls to advance trace events, or select any artifact-provided entry from **Jump to difference**.

The shared **Incoming source token** panel shows the newly observed token at each event. Its time is `observation_ms`: the time the streaming pipeline observes that source token/event. It is a source-observation clock, not a claim of forced-aligned audio word timing. The viewer never exposes a source token beyond the current `source_token_end`.

Each policy card shows its artifact-provided **Current uncommitted source** span and every buffered token. REAL and ZERO can have different buffers after different earlier commits even though they continue hearing the same incoming token. The newest buffered token is emphasized. WAIT still displays the actual accumulating source span; no translation or policy probability is fabricated until inference is available.

Playback scales consecutive `observation_ms` deltas by the selected speed and caps each individual UI delay at **1,500 ms** to keep long source pauses from stalling the replay. Available speeds are `0.5x`, `1x`, `2x`, `4x`, and `8x`. `1x` retains the relative source-clock feel; `4x` and `8x` suit presentation/review.

Use **Pause on policy divergence** to stop immediately after a rendered event where REAL and ZERO decisions differ. It is off in Research mode and turns on when Presentation mode is enabled. It does not pause merely because candidate translations differ.

**Next differing decision** and **Previous differing decision** are calculated from the loaded synchronized traces. Reset returns to event zero.

Use **Presentation mode** to hide numeric causal features and secondary detail while keeping the incoming token, uncommitted source buffers, policy comparison, prepared knowledge, and committed output in view. Keyboard controls are: `Space` play/pause, `Left`/`Right` step, `N`/`P` next/previous differing decision, `B` focus bookmarks, `R` reset, and `D` toggle pause on policy divergence. Shortcuts do not run while a form control is focused.

For a concise advisor or technical-reviewer walkthrough, see [PRESENTATION.md](PRESENTATION.md).

## Scientific Interpretation

REAL and ZERO use the same P3 checkpoint and share one observed source stream. The prepared source remains in provenance for both conditions. ZERO_CONTEXT replaces only the prepared-global policy embedding with exact zeros; it does not remove the source and the prepared document is not supplied to EnViT5 in either condition.

Different policy commit decisions can alter later candidate source boundaries. Any subsequent candidate-translation difference may therefore be a downstream consequence of translating different source spans, not direct prepared-context injection into EnViT5. This controlled DEV illustration does not establish general quality superiority.
