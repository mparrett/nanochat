# Feature: NTK-Mirror vs L1 LoRA cross-adapter comparison on persona-retention

**Filed:** 2026-05-24
**Source:** Today's NTK-Mirror smoke success + composability validation
(see `docs/project_notes/backlog.md` entry).

## Question

Does NTK-Mirror (sparse signed log-gates in activation space) beat or
match L1 LoRA v2 (rank-16 Q+K+V+O weight-space adapter) on our
persona-retention dataset?

**Reference to beat:** L1 LoRA v2 at 2026-05-07 close-out — **19/30
all_three** on held-out (recall name AND role AND location).

## Two implementation paths

### Path A — NTK-Mirror on Qwen2.5-0.5B-Instruct (no graft, ~3-5h)

Use NTK-Mirror in its native habitat. Reuse `persona_retention_v1.jsonl`
(300 train + 30 eval, already curated for nanochat's prompt format —
may need Qwen chat-template adaptation, ~30min).

**Pros:** Cheap; gates NTK-Mirror's mechanism on a small Qwen without
porting work. Directly comparable to the published-paper headline use.

**Cons:** Not apples-to-apples with L1 LoRA v2 (different base, different
tokenizer, different SFT recipe). If NTK-Mirror wins, attribution is
ambiguous between mechanism and base. If it loses, same problem.

**Wall budget:** ~30min dataset adaptation + ~5min fit + ~5min eval +
~1h writeup. With memory cleanup (preflight matters more here than for
nanochat — see backlog wall-pace lesson).

### Path B — Graft NTK-Mirror into nanochat, run on d6_baseline_modern (~1-2d total)

Port the mechanism into `nanochat/gpt.py` (see graft notes below). Then
fit a controller on `persona_retention_v1.jsonl` using the same
`d6_baseline_modern` SFT checkpoint our L1 LoRA v2 ran against.

**Pros:** Apples-to-apples — same base, same data, same eval harness.
Direct test of mechanism (activation-space rescaling vs weight-space
rank-r addition). Plus the graft asset itself becomes reusable: future
bench v0 work could test NTK-Mirror as a third architectural arm
alongside Stage 1 / Stage 2.

**Cons:** ~1d graft effort. Risk that nanochat's custom `Block` + `Linear`
classes don't fit NTK-Mirror's HF-shaped assumptions cleanly without
non-trivial adaptation.

**Wall budget:** ~1d port + ~30min dataset adaptation + ~5min fit + ~5min
eval + ~1h writeup. The port is dominated by debugging time, not LOC.

## Graft notes (for Path B)

NTK-Mirror's `_LAYER_PATHS` in `src/ntkmirror/layers.py` already includes
`"transformer.h"` (GPT-2 pattern) which is what nanochat uses. **Layer
detection likely works out of the box.**

Required mechanism components (~200-400 LOC port to a new
`nanochat/ntk_mirror.py`):

1. **Gate selection:** forward + backward pass to get per-channel
   `dL/ds_{l,c} = sum_t <dL/dh_{l,t,c}, h_{l,t,c}>` over a small scoring
   set. Select top-K by magnitude.
2. **Apply intervention:** per-channel multiplicative gate on Block
   output. Trainable parameter is `s` (signed log-gate), forward applies
   `exp(tanh(s) * max_log_gate)`.
3. **Gate fit:** AdamW over the selected `s` parameters only. Base frozen
   (use `torch.no_grad()` outside the gate-fit path).
4. **Save/load:** sparse dict `(layer_idx, channel_idx, raw_value)`.
   ~mirror the LoRA `state_dict` save/load pattern in `nanochat/lora.py`.
5. **Compose:** dictionary merge + clip. Same algebra as
   `compose_states` in `ntkmirror/compose.py`.

**Existing nanochat infrastructure that helps:**
- The Stage 1.5b `W_o init` audit work means we know nanochat's Block
  boundary intimately.
- The `bench/` runner already shows how to attach memory mechanisms to
  specific blocks.
- `nanochat/lora.py` (~280 LOC) shows the adapter-save/load + apply-walker
  pattern; NTK-Mirror's structure is similar enough that the same shape
  applies.
- `nanochat/gpt.py::Block` is the natural hook target (output of each
  block before residual addition).

**Risks:**
- nanochat uses `RMSNorm` and pre-norm flow; NTK-Mirror's published code
  was tested against post-norm models. May affect gate-selection
  heuristic effectiveness.
- nanochat's `Linear` class manages master-fp32 / matmul-in-x.dtype — the
  intervention is on hidden states, not weights, so should compose
  cleanly, but worth verifying first.

## Pass / fail criteria

- **all_three on held-out persona retention** (30 eval):
  - NTK-Mirror **beats 19/30** → real signal that activation-space is
    superior to LoRA at our scale; consider as the new preferred adapter.
  - NTK-Mirror **below 13/30** → LoRA wins cleanly; document the negative
    result; mechanism shelved at our scale.
  - 13-19 → noisy / inconclusive; need n=2-3 seed bracketing before
    drawing conclusions.
- **Template-bleed check** via the 7-prompt multi-turn rubric (the same
  one L1 LoRA v2 hit 2/7 on with the persona template firing on every
  prompt). Does NTK-Mirror have the same shape problem, a different one,
  or none?
- **Composition follow-up** (only if cross-adapter wins): does adding
  NTK-Mirror's persona controller to a hypothetical other-task
  controller (e.g., self-correction) retain both?

## Pre-launch hygiene

- Memory cleanup BEFORE launch (today's NTK-Mirror disjoint runner hit
  4× slowdown from paging — much worse than nanochat training would).
- `dev/preflight_memory.py` returning exit 0 is the gate.
- Specifically kill `faprox.py` + `server.py` if not load-bearing for
  that session.

## Recommendation

**Path A first** as cheaper validation. If NTK-Mirror beats 19/30 on
Qwen2.5-0.5B-Instruct, **then Path B graft** is worth the ~1d investment
for the apples-to-apples follow-up + the reusable architecture-arm asset.

If Path A loses cleanly (≤13/30), Path B becomes much less attractive —
the mechanism doesn't pay off at this scale regardless of base, and the
graft is more expensive than the signal warrants.

## Outputs to commit (when revisited)

- Markdown writeup: `docs/ntkmirror_persona_comparison_<date>.md`
- HTML narrative writeup: `docs/ntkmirror_persona_comparison_<date>.html`
  (per CLAUDE.md convention: markdown first as insurance, HTML for
  publication style)
- HANDOFF.md addendum

## References

- Backlog entry: `docs/project_notes/backlog.md::NTK-Mirror`
- L1 LoRA v2 reference: `docs/lora_l1_persona_2026-05-07.md` and HTML
- Dataset: `~/.cache/nanochat/persona_retention_v1{,_eval}.jsonl`
- ntkmirror local clone: `~/projects-new/3p/ntkmirror/`
- ntkmirror smoke results: `~/projects-new/3p/ntkmirror/runs/disjoint_composition/`
