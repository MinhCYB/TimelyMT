# TimelyMT — Prepared Context Extension Workplan

**Status:** Planning / staged implementation  
**Scope:** TRAIN + DEV only  
**TEST:** Untouched unless explicitly approved by the researcher  
**Goal:** Extend TimelyMT so genuinely pre-talk prepared material can be represented, validated, and eventually used by a new context-aware policy variant without modifying frozen V1/V2 conclusions.

---

## 0. Working Agreement

This document is the persistent coordination file for the Prepared Context extension.

The working loop is:

1. **Minh + ChatGPT** define the next bounded task and produce a precise Codex prompt.
2. **Codex** inspects/implements only that task.
3. **Codex reports back**:
   - files changed,
   - commands run,
   - tests run,
   - artifacts created,
   - unresolved questions,
   - any deviation from the task.
4. **Minh sends the Codex report back to ChatGPT.**
5. **ChatGPT reviews** the result against the repository architecture, research constraints, and previous decisions.
6. Only after review do we define the next task.

Do not let Codex silently expand scope. Each task should be independently reviewable.

---

## 1. Non-Negotiable Research Constraints

- V1 is frozen and immutable.
- Existing V2 frozen artifacts and conclusions must remain unchanged.
- V2 remains a post-hoc exploratory DEV extension.
- Do not present V2 as outperforming V1.
- Do not retrain existing V2 checkpoints unless explicitly required by a later approved task.
- Do not access or execute TEST.
- Do not inspect TEST references.
- Do not tune thresholds, model design, retrieval rules, or context construction on TEST.
- Prepared material must be genuinely available before the talk.
- Prepared context must not be derived from:
  - complete future transcript,
  - Vietnamese references,
  - alignments,
  - reference-derived glossaries,
  - TEST data,
  - summaries generated from the complete talk transcript.
- Any new experiment must be a new variant / extension and must not silently mutate P0/P1/P2 semantics.

---

## 2. Current Confirmed State

Current V2 uses only causal streaming context:

- current uncommitted source text,
- immediately previous committed source unit,
- immediately previous generated target unit,
- 11 numeric streaming/stability features.

Prepared documents are **not** currently used by:

- EnViT5 translation,
- V1 policy,
- V2 P0/P1/P2,
- training supervision,
- runtime state,
- feature construction.

The current translator input is conceptually:

```text
en: <current uncommitted English source span>
```

The prepared-context path is therefore a new research extension rather than a UI-only feature.

---

## 3. Target Research Direction

Primary research question:

> Does genuinely pre-talk semantic context provide useful information for streaming COMMIT decisions beyond causal streaming history?

The initial intervention should be **policy-side prepared context**, while keeping the translator frozen and unchanged.

Reason:

```text
P2 baseline
=
streaming history
+
same EnViT5
+
same source stream
+
same timing
+
same numeric features

P3 prepared-context variant
=
everything above
+
prepared context representation
```

This makes the independent variable easier to interpret.

Translator conditioning can be investigated later as a separate ablation.

---

## 4. Proposed Architecture

```text
                   BEFORE TALK
                       │
       ┌───────────────┴────────────────┐
       │                                │
   Description                       Slides
   Abstract                          Notes
   Glossary                          Paper
   Other approved material
       │                                │
       └───────────────┬────────────────┘
                       ▼
             PreparedContextPool
                  one per talk
                       │
                freeze / validate
                       │
                context encoding
                       ▼
          PreparedContextRepresentation
                       │
                       │
LIVE STREAM            │
    │                  │
    ├─ current source ─┤
    ├─ previous source ┤
    ├─ previous target ┤
    └─ 11 numeric ─────┤
                       ▼
                  New policy
                       │
                  p(COMMIT)
                  /       \
              LISTEN      COMMIT
```

The `PreparedContextPool` is talk-specific and immutable after the talk begins.

---

## 5. Planned Experiment Progression

### Stage A — Data feasibility

Determine which genuine pre-talk materials are available for the existing 12 TRAIN and 3 DEV talks.

No model changes.

### Stage B — PreparedContext contract

Define a stable, provenance-aware artifact format.

No training yet.

### Stage C — P3-GLOBAL

Create the smallest controlled extension:

```text
P3-GLOBAL =
MiniLM(current source)
+ MiniLM(previous committed source)
+ MiniLM(previous generated target)
+ MiniLM(prepared talk context)
+ 11 numeric features
```

Expected input dimension with 384-d MiniLM embeddings:

```text
384 × 4 + 11 = 1547
```

The prepared representation is fixed for the whole talk.

### Stage D — Evaluate P3-GLOBAL on DEV

Compare against frozen P2 under controlled conditions.

### Stage E — P3-RETRIEVAL

Only if the global-context experiment is scientifically useful.

```text
current source
→ query embedding
→ search PreparedContextPool
→ top-k relevant chunks
→ aggregate context representation
→ policy
```

### Stage F — Research Demo

After an actual prepared-context model exists, build the UI that visualizes:

- MP3 playback,
- word emission over time,
- current source buffer,
- candidate translation,
- policy state,
- p(COMMIT),
- threshold,
- WAIT / LISTEN / COMMIT,
- prepared context pool,
- retrieved / active context,
- committed Vietnamese output,
- side-by-side baseline vs prepared-context policy behavior.

---

## 6. Phase A — Prepared Material Inventory

### Objective

For every TRAIN and DEV talk, determine what prepared information is genuinely available before the talk.

### Candidate sources

- talk title,
- speaker,
- domain/topic,
- official TED/public description,
- abstract,
- public slide deck,
- speaker notes,
- linked paper,
- public project page,
- glossary / terminology,
- other public pre-talk material.

### Classification

Every source must be classified as one of:

```text
SAFE_PRETALK
QUESTIONABLE
TRANSCRIPT_DERIVED
REFERENCE_DERIVED
UNAVAILABLE
```

### Required output

```text
| talk_id | split | safe sources | questionable sources | unavailable | notes |
```

And totals:

- TRAIN talks with usable context,
- DEV talks with usable context,
- source type available consistently across the dataset,
- whether description/title/domain are enough for an initial P3-GLOBAL experiment.

### Completion gate

Do not proceed to implementation until we know what real data P3 will encode.

---

## 7. Phase B — PreparedContext Data Contract

Proposed conceptual format:

```text
PreparedContextPool
├─ schema_version
├─ talk_id
├─ split
├─ frozen_at
├─ sources[]
│  ├─ source_id
│  ├─ source_type
│  ├─ source_uri / artifact identity
│  ├─ acquired_at
│  ├─ available_before_talk
│  ├─ language
│  ├─ checksum
│  ├─ extraction_method
│  ├─ transcript_used
│  └─ reference_used
└─ chunks[]
   ├─ chunk_id
   ├─ source_id
   ├─ text
   └─ metadata
```

Embedding storage should be decided separately after the raw contract is stable.

### Required validators

At minimum:

- talk ID match,
- split match,
- source checksum,
- pre-talk availability assertion,
- no target/reference source,
- no future transcript source,
- no TEST-derived material,
- deterministic chunking,
- provenance completeness.

### Completion gate

The same context artifact must be loadable deterministically without touching the canonical target reference or future source stream.

---

## 8. Phase C — P3-GLOBAL Design

### Control

Frozen P2 semantics remain unchanged.

### New variant

Use a new explicit variant name. Do not redefine P2.

Working name:

```text
P3_GLOBAL
```

Final naming can be changed before implementation.

### Feature shape

```text
current_source_embedding             384
previous_committed_source_embedding  384
previous_generated_target_embedding  384
prepared_context_embedding           384
numeric_features                      11
-----------------------------------------
total                               1547
```

### Prepared representation

For the first experiment:

- build one deterministic talk-level prepared text representation,
- encode with the same frozen multilingual MiniLM,
- compute once per talk,
- reuse at every policy decision,
- do not perform dynamic retrieval yet.

### Training consequence

Existing V2 checkpoints cannot consume the extra 384 dimensions.

P3 requires a new checkpoint and corresponding metadata.

### Keep constant

- source streams,
- TRAIN/DEV split,
- translator model and revision,
- EnViT5 decoding,
- streaming min/max unit constraints,
- 11 numeric features,
- policy hidden architecture where possible,
- optimizer/training schedule where possible,
- threshold grid,
- evaluation metrics,
- DEV selection protocol,
- causal rollout semantics.

---

## 9. Phase D — Evaluation Plan

Primary comparison:

```text
P2
vs
P3-GLOBAL
```

Quantitative metrics:

- BLEU,
- chrF2,
- AL,
- LAAL,
- number of commits,
- commits / 100 source tokens,
- forced commit rate,
- first commit source tokens,
- first commit simulated source-clock latency,
- average source tokens per unit.

Additional analysis required for the demo:

### Decision divergence

Record cases where both policies see the same source clock but behave differently.

Examples:

```text
P2: p(COMMIT)=0.37 → LISTEN
P3: p(COMMIT)=0.62 → COMMIT
```

and:

```text
P2: COMMIT
P3: LISTEN
```

For each divergence, preserve enough state to inspect:

- current candidate source,
- current candidate translation,
- previous committed source,
- previous generated target,
- prepared context representation/source,
- numeric features,
- threshold,
- p(COMMIT),
- final action.

This qualitative trace is important for the research demo.

---

## 10. Phase E — P3-RETRIEVAL (Deferred)

Do not implement until P3-GLOBAL is reviewed.

Concept:

```text
PreparedContextPool
      ↓ chunk + embed once
Current source
      ↓ query embedding
cosine similarity
      ↓
Top-K context chunks
      ↓
aggregate
      ↓
P3 retrieval policy
```

Questions to decide later:

- chunk size,
- overlap,
- Top-K,
- similarity threshold,
- aggregation method,
- whether glossary items use a separate representation,
- query = current source only or source + history,
- retrieval caching,
- retrieval provenance in prediction traces.

All of these are research decisions and must not be tuned on TEST.

---

## 11. Demo Scope

The demo has one purpose:

> Show what the streaming translator and policy are doing, and make differences between policies understandable to a human observer.

No:

- authentication,
- database,
- project management,
- report export,
- general dashboard,
- SaaS functionality.

### Core UI

```text
┌──────────────────────────────────────────────┐
│ PREPARED KNOWLEDGE                           │
│ context available / active context           │
├──────────────────────────────────────────────┤
│ AUDIO                                        │
│ playback + synchronized word emission        │
├──────────────────────────────────────────────┤
│ LIVE SOURCE STREAM                           │
│ heard tokens vs future tokens                │
├──────────────────────────────────────────────┤
│ POLICY A               POLICY B              │
│ candidate              candidate             │
│ p(COMMIT)              p(COMMIT)             │
│ threshold              threshold             │
│ WAIT/LISTEN/COMMIT      WAIT/LISTEN/COMMIT    │
├──────────────────────────────────────────────┤
│ POLICY INPUT / CONTEXT                       │
├──────────────────────────────────────────────┤
│ TRANSLATION EVOLUTION                        │
├──────────────────────────────────────────────┤
│ LIVE VIETNAMESE OUTPUT                       │
└──────────────────────────────────────────────┘
```

### Required state distinction

```text
WAIT
= fewer than minimum source tokens;
  policy inference has not happened.

LISTEN
= policy inference happened;
  p(COMMIT) is below threshold.

COMMIT
= policy inference happened and crossed threshold,
  or a forced boundary applied.
```

Never display `p(COMMIT)=0` for the WAIT state if the policy has not actually been evaluated.

---

## 12. Demo Runtime Architecture

The UI may compare two policies, but each policy must have its own independent streaming session.

```text
                 shared audio/source clock
                          │
               ┌──────────┴──────────┐
               ▼                     ▼
          Session A              Session B
          own start              own start
          own commits            own commits
          own history            own history
               │                     │
          PolicyAdapter          PolicyAdapter
               │                     │
               └──────────┬──────────┘
                          ▼
                shared translator/cache
                          │
                     frozen EnViT5
```

Do not share commit/history state between policies.

A COMMIT changes future segmentation and history only for that policy.

---

## 13. Codex Task Rules

Every Codex task should:

1. inspect before modifying,
2. state assumptions,
3. preserve current frozen artifacts,
4. avoid TEST,
5. avoid unrelated refactors,
6. add/update tests for new behavior,
7. run only relevant safe tests,
8. report exact commands executed,
9. report exact files modified/created,
10. stop if repository evidence contradicts the task assumptions.

### Mandatory Codex handoff format

Every implementation task should end with:

```markdown
# Codex Handoff

## Task Completed

## Files Changed

## Architecture / Behavior Added

## Tests Added or Updated

## Commands Run

## Results

## Existing Artifacts Preserved

## TEST Safety

## Deviations From Prompt

## Open Questions

## Recommended Next Step
```

This handoff should be sent back to ChatGPT for review before continuing.

---

## 14. Work Queue

### TASK 01 — Prepared Context Inventory

**Status:** NEXT

Goal:
Determine what genuine pre-talk material exists for all TRAIN and DEV talks.

Output:
Inventory report only.

No code changes.

---

### TASK 02 — Review Inventory

**Owner:** Minh + ChatGPT

Decide:

- allowed prepared sources,
- minimum source coverage,
- whether description/title/domain are sufficient,
- whether additional public documents are necessary.

---

### TASK 03 — Define PreparedContext Schema

**Status:** BLOCKED BY TASK 02

Implement data contract + validators only.

No policy/model changes.

---

### TASK 04 — Build TRAIN/DEV PreparedContext Artifacts

**Status:** BLOCKED BY TASK 03

Create deterministic prepared-context artifacts with full provenance.

No TEST.

---

### TASK 05 — P3-GLOBAL Design Review

**Status:** BLOCKED BY TASK 04

Before implementation, inspect exact integration points and produce a code-change plan.

---

### TASK 06 — Implement P3-GLOBAL

**Status:** BLOCKED BY TASK 05

Add new variant without changing P0/P1/P2.

---

### TASK 07 — Train / DEV Rollout

**Status:** BLOCKED BY TASK 06

Train only the new approved variant and evaluate on DEV.

Do not run TEST.

---

### TASK 08 — Scientific Review

**Owner:** Minh + ChatGPT

Review quantitative results and decision-divergence traces.

Decide whether P3-RETRIEVAL is justified.

---

### TASK 09 — P3-RETRIEVAL

**Status:** DEFERRED

Only proceed if TASK 08 supports it.

---

### TASK 10 — Research Demo

**Status:** DEFERRED UNTIL A REAL PREPARED-CONTEXT VARIANT EXISTS

Implement the minimal audio-synchronized policy visualizer.

---

## 15. Current Next Action

Run **TASK 01 — Prepared Context Inventory**.

After Codex finishes, save or paste its handoff/report and review it with ChatGPT.

Do not proceed automatically to TASK 03.

The next implementation decision depends on what genuine pre-talk material actually exists.
