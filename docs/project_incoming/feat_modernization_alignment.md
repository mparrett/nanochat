# Feature: 2026 LLM-template alignment audit

**Filed:** 2026-05-05
**Source:** Cross-project synthesis with trx4mr's `incoming-synthesis-2026-05-05.md`
(Stanford CS336 Lecture 3 on 2026 LLM architecture conventions, via Tatsu).
trx4mr ran the same audit against picoGPT and filed ADR-019 (z-loss
standby).

## Context

CS336 Lecture 3 is the public framing of "what 2026 production LLMs
have converged on." The lecture's claim is that 90% of architecture
choices are now stock; you can copy any open-source mainstream model
and they're virtually identical along the dimensions covered.

Audit summary against nanochat: **~85% on-template.** Three deliberate
departures and three concrete additions worth considering. This ticket
catalogs both.

## What we already match (no action needed)

| Convention                       | nanochat status                             |
| -------------------------------- | ------------------------------------------- |
| Pre-norm (LN outside residual)   | ✅ `norm(x)` before attn + mlp              |
| RMSNorm (no mean, no bias)       | ✅ `F.rms_norm` (`gpt.py:75`)               |
| No bias terms                    | ✅ every `Linear(..., bias=False)`          |
| RoPE                             | ✅ `apply_rotary_emb` (`gpt.py:125`)        |
| Serial transformer blocks        | ✅                                          |
| LN sprinkling                    | ✅ `norm(x)` at multiple boundaries         |
| GQA                              | ✅ via `n_kv_head` config                   |
| Alternating local + global attn  | ✅ **`window_pattern="SSSL"`** default      |
| FFN 4× expansion (non-GLU)       | ✅ matches non-GLU rule                     |
| heads × head_dim ≈ hidden_dim    | ✅ 6 × 64 = 384 at d6                       |
| Vocab in monolingual range       | ✅ 32,768                                   |
| Weight decay as optimizer lever  | ✅ `weight_decay=0.28`                      |
| Logit soft cap                   | ✅ `softcap=15` (`gpt.py:741-745`)          |

That's a strong baseline. Notably the alternating local/global
attention via `window_pattern="SSSL"` is exactly the Llama 3 / Gemma 2
/ Olmo 2 pattern, and the logit soft cap is the Gemma-only trick
already wired in.

## Deliberate departures (probably keep)

### 1. ReLU² instead of SwiGLU/GeGLU

**Where:** `gpt.py:200-205`. `c_fc(x)` then `F.relu(x).square()`.

**Lecture claim:** SwiGLU/GeGLU is universal in modern open models;
performance difference is "negligible — pick either."

**Why nanochat differs:** ReLU² is competitive with GLU at slightly
less compute and far simpler (one Linear instead of gated pair). Per
papers like Primer, Pufferfish, this is a paper-supported choice, not
a regression. Aligns with nanochat's "strong baseline, not a
configurable framework" stance.

**Recommendation:** keep as the default. Worth one sweep at a chosen
depth (e.g. d6 or d8) to confirm the "negligible" claim holds at our
scale before any future deeper investment.

### 2. Aspect ratio = 64 (vs lecture's ~100)

**Where:** `--aspect-ratio` CLI default, `n_embd = depth × aspect_ratio`.

**Lecture claim:** `hidden / n_layers ≈ 100` is the production sweet
spot balancing pipeline-parallelism + expressivity.

**Why nanochat differs:** the depth-sweep philosophy (`--depth` is the
single complexity dial) requires a fixed aspect ratio so that any
depth produces a cleanly comparable model. 64 was the calibration
point. Width-vs-depth re-tuning per scale would defeat the discipline.

**Recommendation:** keep. Document the deviation so future architecture
discussions know it's deliberate, not an oversight. trx4mr/picoGPT
hit the same trade for the same reason.

### 3. Logit softcap present, z-loss + qk_norm absent

**Where:** `gpt.py:741-745` (softcap=15); no qk_norm or z-loss in code.

**Lecture claim:** z-loss and qk_norm are "more universal" stability
tricks than logit softcap, which is "Gemma-only, costs perf, use
cautiously."

**Why nanochat differs:** unclear. Possibly intentional (nanochat is
small enough that softcap alone is sufficient, the others' overhead
isn't worth it at our scale), possibly just convention from the
codebase's lineage. Worth investigating but not a bug — at d6 we've
seen no spike events that would have demanded z-loss / qk_norm
intervention.

**Recommendation:** keep softcap; add z-loss + qk_norm only conditional
on instability evidence at higher depth (see additions below).

## Additions worth considering

### A1: Z-loss as instability standby

**What:** `(log Z)²` regularizer on the softmax normalizer to keep `Z`
near 1, preventing softmax-explosion-induced loss spikes.

**Where it's used in production:** DCLM, Olmo. Standard "first thing
to reach for" if a 1-bit / ternary / aggressive-quantization run starts
spiking.

**Cost:** ~5 lines, near-zero compute overhead.

**When to add:** if d=8 extension or future quantization work shows
loss spikes that softcap doesn't suppress. Park as an ADR in
`decisions.md` documenting "z-loss is the standby stability lever" so
future Claude sessions know the option exists. trx4mr's ADR-019 takes
exactly this stance — adopt the same mental model here.

**Acceptance:** if added, run a control comparison at iso-config with
and without z-loss to confirm no regression on stable runs (it
shouldn't, but verify).

### A2: QK norm

**What:** RMSNorm applied to Q and K each before the QK matmul. Keeps
softmax inputs at unit scale across training.

**Where it's used in production:** "all big models add it" per the
lecture; originated in multimodal community.

**Cost:** ~3 lines (two extra `norm()` calls in `forward`), negligible
compute. Some papers report 0.5-1% val_bpb improvement at scale.

**When to add:** worth a sweep at d=8 specifically — more layers = more
chances for QK softmax to drift. If d=8 extension activates, a
qk_norm A/B is a cheap addition to that experiment plan. **Important:
adding qk_norm changes the model — must re-baseline (per recipe-drift
lesson from track synthesis).** Cannot retroactively compare against
existing d6 numbers.

**Acceptance:** at d=8, qk_norm should be neutral-or-better on val_bpb
at iso-params. If neutral, keep for the safety; if better, becomes the
default.

### A3: SwiGLU/GeGLU sweep at iso-params

**What:** swap `F.relu(x).square()` for SwiGLU (`F.silu(W₁x) * W₂x`),
scale `d_ff` from 4× to 8/3× to hold params constant.

**Why:** the only way to test the lecture's "negligible" claim at our
scale. Currently we're guessing based on Primer-era papers. A direct
comparison closes the question.

**Cost:** ~10 lines (new MLP variant + a CLI flag), one full pretrain
each side at chosen depth (~3-5h each). Total ~6-10h.

**When to add:** if d=8 extension activates and Phase 2 has compute
budget for it, slot in alongside as an additional condition. Otherwise
defer indefinitely — at d6 the "boring 2026 stack" question is less
important than the architectural-question completeness.

**Acceptance:** if SwiGLU comes in materially better (Δ val_bpb
> 0.005, larger than seed-noise), it becomes the default and ReLU² is
deprecated. If neutral within seed-noise, keep ReLU² for simplicity.

## Priority order

If picked up:

1. **A1 (z-loss as standby ADR)** — zero-cost documentation move.
   Mirrors trx4mr's ADR-019. Just adds the lever to project memory so
   future sessions know it exists. Do this regardless.
2. **A2 (qk_norm at d=8)** — only if `feat_d8_extension.md` activates.
   Slot into Phase 1 derisk + Phase 2 headline as a free A/B.
3. **A3 (GLU sweep)** — defer unless track is otherwise revived. The
   architectural question we'd answer ("does the lecture's
   'negligible' claim hold at d6/d8?") is less consequential than the
   d=8 memory-architecture question itself.

## What's not in scope

- **Add general "modernization framework" with feature flags.**
  nanochat is "a strong baseline, not a configurable framework" per
  CLAUDE.md. Each addition should be either (a) the new default if it
  wins, or (b) deferred. No `--use-swiglu` flag if SwiGLU just becomes
  the default.
- **Aspect ratio re-tuning across depth.** Defeats the depth-sweep
  discipline. Documented as deliberate.
- **Long-context attention beyond `window_pattern="SSSL"`.** SSM
  variants (Qwen 2.5's swap), Cohere Command R alternation. Out of
  scope unless we extend max_seq_len materially.
- **Tokenization / vocab modernization.** Out of scope — nanochat's
  existing tokenizer is fit-for-purpose at our scale.

## Cross-project synchronization

trx4mr filed the same audit against picoGPT and adopted z-loss as a
standby (ADR-019). If we adopt A1, mirror trx4mr's ADR pattern in
`docs/project_notes/decisions.md` so the rationale is symmetric across
the two projects. The `MEMORY.md` `feedback_codex_collaborator.md` lens
also applies — these "modernization" suggestions are best filtered
through the lecture's own framing of "negligible vs material" before
adoption.
