# Backlog

Directions the operator wants to try but has not scheduled. Newest first.
Each entry: short pitch, rough cost, why it's interesting, pointer to the
literature or precedent. Move to `decisions.md` if/when picked up.

---

## NTK-Mirror: LoRA-free activation-space adapter, composability validated (2026-05-24)

**Pitch.** Chlon (Hassana Labs), `leochlon/ntkmirror`, paper forthcoming.
Frozen HF causal LM + sparse signed log-gates on decoder-layer output
channels: `h' = exp(s) · h` with `|s| ≤ max_log_gate` (default 0.05).
Gates selected by `|dL/ds|` (NTK-flavored), fit by AdamW. Default 5000
gates / 240 steps / lr 5e-3. Compose via gate-space addition (additive
in log-gate space ⇔ multiplicative in activation space). Persistent-
memory store with retrieve+compose+attach. Same use-case envelope as
MeMo at ~100,000× less compute.

**Validated on M2 (smoke + composability runner, 2026-05-24).** Two runs
on Qwen2.5-0.5B-Instruct via MPS, no source modifications required:

- **Single-task smoke** (512g/40steps, math demo): loss 1.764 → 1.642,
  "47 + 36 = ?" → "83" (correct).
- **Disjoint composition** (5000g/240steps, GSM8K + MBPP, 64/32 train/eval):

| controller | gsm8k NLL | mbpp NLL |
|---|---:|---:|
| base | 0.714 | 1.064 |
| gsm8k | **0.600** | 1.058 |
| mbpp | 0.736 | **0.813** |
| **composed** (gsm8k + mbpp) | **0.614** | **0.911** |

Composed controller retains **88% of GSM8K single-task gain and 61% of
MBPP**. No catastrophic interference, no regression below base.
Composition report: 5000 gates each, 80% gate overlap (Jaccard 0.66),
near-orthogonal cosine (0.024) — controllers select the same channels
but assign nearly orthogonal signed values, which is the structural
condition the composability claim requires.

**Why it matters to us.**
- **Direct alternative to LoRA at our scale.** Our L1 LoRA v2 (rank-16
  Q+K+V+O on d6 nanochat) hit a "one-pattern adapter" ceiling. NTK-Mirror
  occupies a different design space — activation-space rescaling vs
  weight-space rank-r additions. Direct head-to-head testable.
- **Composability story extends.** Our LoRA work didn't test
  task-arithmetic compositionality. NTK-Mirror's gate-space addition
  has a cleaner algebraic justification than LoRA weight-addition,
  and now empirically holds at this scale.
- **Different point on the parameters-vs-adaptability spectrum** than
  Hope/NL Stage 2 (intrinsic mechanism), δ-mem (recurrent state), and
  MeMo (full second LLM).

**Wall-pace lesson.** Memory pressure dramatically affects MPS step time
(this run: ~42 s/step at 0.5 GB free → ~21 s/step after operator killed
some background apps). For longer NTK-Mirror experiments on M2: aggressive
memory preflight matters more than for nanochat training where optimizer
compute dominates and paging effect is smaller.

**Engagement shapes** (cheapest first):
1. **Backlog only** — current state. Composability validated; cross-adapter
   ran and landed at parity; no further experiment justified by the result.
2. **Cross-adapter comparison on persona-retention** (Path A — ✅ done
   2026-05-25). Result: mean 18/30 all_three across n=3 seeds (range 17-20)
   on Qwen2.5-0.5B-Instruct, **parity** with L1 LoRA v2's 19/30 on
   `d6_baseline_modern_sft`. Writeup:
   `docs/ntkmirror_persona_comparison_2026-05-25.md`.
3. **Graft to nanochat** (Path B, ~1d). **Shelved** per the cross-adapter
   ticket's pass criteria — parity result doesn't justify the engineering.
   Reopen if a downstream need (composability, persistent-memory store,
   sparse-dict adapters as a bench v0 third arm) makes the graft asset
   worth building for its own sake.

**Cost.** 0 (default; cross-adapter done and shelved).

**Status.** Closed-but-reopenable. Smoke validation done; cross-adapter
landed at parity (writeup committed); Path B graft shelved unless
downstream demand surfaces. Upstream-PR ticket still open if revisited.

**References.**
- Repo: `https://github.com/leochlon/ntkmirror` (MIT, paper forthcoming).
- Local clone: `~/projects-new/3p/ntkmirror/` (main branch, fresh clone;
  local patch on `scripts/run_disjoint_composition.sh` — see
  `feat_ntkmirror_upstream_bash_pr.md`).
- Outputs on disk: `~/projects-new/3p/ntkmirror/runs/disjoint_composition/`
  (2 controllers + composed + 8 eval JSONs + composition_report.json).
- Cross-reference: MeMo backlog entry (different mechanism, much heavier
  compute envelope, same use-case shape).

---

## MeMo: parametric LLM-as-memory framework (2026-05-21)

**Pitch.** Quek, Lee, Leong et al., **MeMo: Memory as a Model**, arXiv:2605.15156v2,
May 2026 (NUS / A*STAR / Tokyo / Liquid AI / MIT CSAIL / AI Singapore / SMART).
A *categorically different* take on "memory" from anything in our active axis:
the memory is **an entire second LLM** (Qwen2.5-1.5B or 14B-Instruct) SFT'd to
internalize a target corpus, queried by a frozen Executive (Qwen2.5-32B or
Gemini-3-Flash) through a 3-stage multi-turn protocol (Grounding → Entity ID →
Answer Synthesis). Two real technical contributions: (1) a 5-step reflection-QA
synthesis pipeline whose Step 5 (cross-document synthesis) does the heavy
lifting — ablating it collapses NarrativeQA from 24% → 6.37%; (2) the structured
inference protocol where Memory responses are compact natural-language
snippets, so retrieval cost is constant in corpus size.

**Why it matters to us.**
- **Matches the ADR-008 pivot direction** ("continuous-learning components
  grafted onto pretrained 4-8B models running locally") exactly — except the
  paper's reference design assumes H100/H200 training (90-180 GPU-h per Memory
  model SFT, 150-240 GPU-h for data synthesis, 240 GPU-h for K=10 merging).
- **Uses our hardware envelope's models as Memory.** The Memory model size in
  their ablation (Qwen2.5-1.5B-Instruct) is roughly Bonsai-1.7B-class. Their
  size-scaling table (Tab 4) shows 1.5B Memory loses meaningfully to 14B Memory
  on all three benchmarks but is non-trivially functional.
- **Different question from δ-mem / Hope/NL Stage 2.** Those are intrinsic
  per-token state inside one model's forward pass; MeMo is external
  composition of two models. Not competing approaches to the same question.
  But MeMo's results suggest the long-context cross-document axis (where we
  found δ-mem and Hope/NL Stage 2 both flat or null at HotpotQA) may need a
  fundamentally different architectural shape than learned-gate recurrent
  state — sub-query decomposition at the protocol level instead.

**Headline numbers** (from Tab 2):

| | BrowseComp-Plus | NarrativeQA | MuSiQue |
|---|---:|---:|---:|
| HippoRAG2 (best RAG) | 56.11 | 21.39 | 42.17 |
| MeMo (Qwen2.5-32B exec) | 54.22 | 26.85 | 48.30 |
| MeMo (Gemini-3-Flash exec) | **66.67** | **53.58** | **60.20** |
| Perfect Retrieval (oracle) | 79.67 | 51.42 | 62.83 |

Beats every RAG baseline on cross-document synthesis tasks; on NarrativeQA
with Gemini-3-Flash Executive, **MeMo exceeds Perfect Retrieval** — the
synthesized reflections carry information the raw evidence docs don't, when
read with a strong reasoner. Also clean noise robustness vs RAG (5-6pp drop
for RAG with 1×N distractors, MeMo ±2pp).

**Why this is not an immediate experiment.**
- **Full reproduction is CUDA-bound.** Step 5 of the synthesis pipeline is
  O(k · C² · Q²); training is 3-epoch SFT on 600k-1.6M QA pairs with
  FlashAttention 2 + DeepSpeed at LR 2e-5 on H100/H200. None of that runs on M2.
- **Not on our active bench v0 axis.** bench v0's question is "do intrinsic
  memory mechanisms specialize on selective vs uniform demands?" — concrete,
  cheap, ~250 min/experiment. MeMo answers a different question.
- **No published Memory model checkpoints.** The MEMORY models are corpus-
  specific (one per BrowseComp-Plus / NarrativeQA / MuSiQue) and the paper
  doesn't release them, so inference-only on a pretrained checkpoint isn't an
  option the way it was for δ-mem.

**Three viable engagement shapes if/when picked up.**
1. **Read-only.** Log to backlog (this entry), no action. *Default and
   probably correct unless a specific corpus motivates engagement.*
2. **Inference-protocol-only** (~1 day). Skip both the synthesis pipeline and
   the SFT. Use an existing Qwen3-4B-8bit or Bonsai-4B as both Memory and
   Executive on a small test corpus; implement the 3-stage protocol in
   `chat_eval_mlx.py`. Tests whether the *protocol* (sub-query decomposition
   + entity narrowing + multi-turn synthesis) contributes anything orthogonal
   to RAG even without trained reflections. Strong negative-result candidate:
   if the protocol alone wins nothing, the SFT-on-reflections is doing the
   work, which sharpens what MeMo really is.
3. **Subset-pipeline** (~2-3 days). Steps 1-3 of synthesis are cheap (run via
   API on a small corpus). Skip Step 5 (the expensive cross-document step),
   accept the corresponding capability ceiling — Tab 9 says removing Step 5
   drops NarrativeQA to 6.37 % — try SFT on a small Memory model on M2 using
   the partial reflections. Cost-controlled capability-degraded reproduction.

**Cost.** Engagement-dependent: 0 (default) / ~1 day (protocol-only) /
~2-3 days (subset-pipeline) / weeks (full reproduction, would need CUDA box).

**Status.** Open. Backlog only. Operator chose "log this for later" on
2026-05-21 after reading the paper; no immediate experiment scheduled.

**References.**
- arXiv:2605.15156v2 (May 2026). Paper file at
  `~/Library/Mobile Documents/com~apple~CloudDocs/Downloads/mem-2605.15156.pdf`.
- Cited Cartridges (ref 65, `arXiv:2506.06266`) as the closest existing baseline
  to MeMo; Cartridges scored 0.00 / 3.75 / 8.57 on Tab 2 (i.e., effectively
  fails the cross-document synthesis question).
- Cross-reference for our work: δ-mem field result
  (`docs/delta_mem_field_result_2026-05-18.html`), strategic pivot ADR-008
  (`docs/strategic_pivot_2026-05-13.md`), bench v0 result HTML
  (`docs/bench_v0_result_2026-05-20.html`), project map
  (`docs/project_map_2026-05-20.html`).

---

## ~~δ-mem reproduction on fp-Qwen3-4B-Instruct~~ — MOVED TO INCOMING (2026-05-17)

**Status:** Promoted from backlog to open ticket on 2026-05-17 after
verifying that public code exists (CC-BY-4.0, real implementation),
pre-trained adapter is published, and operator wants to pursue.

**Active ticket:** `docs/project_incoming/feat_delta_mem_mlx_port.md`
— full reproduction plan (Path B MLX port preferred, Path A PyTorch+MPS
fallback). Self-contained enough to bootstrap a fresh session.

Original backlog entry preserved below for context on how the
direction came together.

---

## δ-mem reproduction on fp-Qwen3-4B-Instruct (2026-05-17, original backlog entry)

**Pitch.** Reproduce δ-mem (Lei et al., May 2026 — `docs/paper_delta_mem_2026-05-17.md`)
on the fp-Qwen3-4B-Instruct backbone we've now canonically baselined.
The architecture is essentially what we paused on 2026-05-13: graft a
small trainable memory module onto a frozen Qwen3-4B base, train via
standard SFT, evaluate via our existing harness. Mechanism is a
gated delta-rule on an 8×8 associative-memory state that produces
low-rank corrections to the frozen attention's query and output.

**Why now / why interesting.**
- The architecture matches our parked Hope/NL graft direction; if a
  published reference exists, we're not reinventing.
- The Stage 2 work on Hope/NL built the exact same gated delta-rule
  from scratch — `nanochat/gpt.py` has reusable building blocks.
- The fp-Qwen3-4B-8bit baseline (ChatCORE 0.7656 at -x 200) gives us
  a precise reference point for measuring lift on our six tasks.
- The δ-mem paper validates on long-context and conversational-memory
  tasks (LoCoMo, MemoryAgentBench) which are NOT our ChatCORE suite.
  Adding our ARC/MMLU/GSM8K/HumanEval/SpellingBee numbers would be
  net signal beyond their published evals.

**Cost.** Three sub-stages, gateable:
- **(a) Verify code availability** (~30 min). Check Declare-lab and
  MindLab-Research GitHub orgs for runnable code. If yes → (b). If
  no → from-scratch reimplementation is ~3-5 days and the gating
  question is whether that's worth it without their training data.
- **(b) PyTorch + MPS smoke** with their adapter wired to fp-Qwen3-4B-
  Instruct (~half day). Verify backward flows correctly through the
  rank-r adapters only, not the frozen backbone.
- **(c) Train on a small SFT sample** (~1-2 days). Their training
  setup uses LongSFT-3; would need to identify the exact mixture.
  Evaluate via our chat_eval_mlx for direct comparison to the
  baselined 0.7656 ChatCORE.

Total feasibly-bounded: 2-3 weeks of focused work; gateable at each
sub-stage; first 30 minutes (the GitHub check) determines whether
the rest is even on the table.

**Falsification thresholds (if the full reproduction happens).**
- ChatCORE on fp-Qwen3-4B-8bit + δ-mem ≥ 0.78: the graft delivers
  measurable lift on our six tasks. Strong signal for the architecture.
- ChatCORE = 0.7656 ± noise: graft works mechanistically (per their
  benchmarks) but doesn't help our reasoning/knowledge suite. Expected
  outcome based on their published numbers; would confirm their lift
  is conversational-memory-specific.
- ChatCORE < 0.7656: the graft hurts. Would falsify the "memory module
  is free to add" assumption.

**What's known so far** (from `docs/paper_delta_mem_2026-05-17.md`):
- Architecture spec is precise enough to reimplement.
- Trainable params are small (rank-r adapters + 8×8 state per layer).
- They report +4.87 average ChatCORE-equivalent lift, +9.31 on
  MemoryAgentBench, +6.13 on LoCoMo. IFEval and GPQA-Diamond flat.
- The "Github: Declare-lab & MindLab-Research" line in the paper
  masthead is a citation, not verified-public runnable code.

**Status.** Open. δ-mem-on-fp is the cleaner experiment; δ-mem-on-int2-
Bonsai may fight the wrong battle (the quantization-cost analysis
shows int2 already preserves reasoning for free; the memory/knowledge
gap is where int2 loses, and that's also where δ-mem helps — so
"compose them" may double the int2 deficit rather than fix it).

**References.**
- Paper notes: `docs/paper_delta_mem_2026-05-17.md` (the precise
  mechanism, reservations, and what this changes for us)
- Strategic context: `docs/strategic_pivot_2026-05-13.md` (why this
  direction was parked, why it now has a reference architecture)
- Backbone baseline: `docs/qwen3_4b_quantization_cost_2026-05-17.md`
- Prior from-scratch precedent: `docs/hope_nl_stage2_*.md` (we built
  the gated delta-rule before, but as a full block replacement; δ-mem
  uses the same equation as a graft module instead)

---

## bf16 full-pretrain validation on M2 — de-risk before any 1-bit work (2026-05-07)

**Pitch.** Run a full 5000-iter d6 pretrain with `NANOCHAT_DTYPE=bfloat16`
on M2 and compare val_bpb against the fp32 baseline of **1.174 at step
5000** (canonical `d6_baseline_modern`, documented in
`docs/hope_nl_stage1_full_pretrain_2026-05-01.md` and HANDOFF.md). The
bf16 patch landed 2026-04-30 (commit `7e21999`, upstream PR
`karpathy/nanochat#741`) and works end-to-end functionally — but the
full-horizon A/B against fp32 was deferred and never run. This closes
that loop.

**Why now.** Direct prerequisite to the 1-bit-from-scratch direction
above. If bf16 — a much milder precision reduction than binary — already
shows meaningful val_bpb drift at the 5000-iter horizon, the binary
direction's odds get worse and we want to know that *before* sinking
infra time. Conversely, if bf16 lands within ~0.5 % of fp32, that's a
real precedent for "reduced precision on this hardware preserves quality"
and the binary infra investment becomes easier to justify.

**What's known so far** (from `bugs.md:166`):
- Functionally works end-to-end on M2 MPS post-fix.
- ~3 % throughput change (M2 has no bf16 hardware; M3+ does).
- Memory unlock is real — `device_batch_size=48` works at bf16, OOM at fp32.
- ~1e-3 numerical drift vs fp32 by step 9 — but not measured at horizon.

**Cost.** ~3 h M2 wall (matches the fp32 baseline). $0. The dataset is
already on disk; the recipe is the canonical `runcpu.sh`-shaped run.

**Falsification thresholds.**
- val_bpb ≤ 1.180 at step 5000: bf16 is a free quality preserver on M2;
  binary direction's prior strengthens.
- val_bpb 1.180–1.200: small drift, bf16 usable for non-baseline experiments
  but fp32 stays the canonical reference; binary needs careful validation.
- val_bpb > 1.200: meaningful drift; numerical accumulation matters at this
  scale; binary infra needs a much more careful design (likely Bonsai's
  proprietary method, not vanilla STE).

**Status.** Open. Cheapest experiment in this backlog.

**References.**
- `docs/project_notes/bugs.md:166` — bf16 fix history.
- `HANDOFF.md` Phase 3 step 1 (~line 161) — bf16 audit.
- Commit `7e21999` — the load-bearing patch.
- Upstream PR `karpathy/nanochat#741`.

---

## Reverse-engineer Bonsai's native-1-bit training method (2026-05-07)

**Pitch.** Bonsai (PrismML) claims a "native 1-bit training method" that
is *not* the standard straight-through estimator. The training code is
proprietary; only the whitepaper and Apache-licensed weights are public.
Read the whitepaper carefully, study the open weights' parameter
distributions for clues, and try to reproduce the method (or a credible
hypothesis of it) in a tiny standalone scratch repo or in trx4mr's
blabberverse.

**Why interesting.** If it works as claimed, native-1-bit training
would change the cost calculus of the 1-bit-d6-from-scratch direction
substantially — STE is the assumed fallback there, and Bonsai's method
might give meaningfully better gradient flow at d6 scale.

**Why probably out of scope.** Operator's expectation is that whatever
Bonsai does likely doesn't fit M2 (specialized accelerator-friendly
math, possibly TPU-specific). And reverse-engineering proprietary
training methods is a research-paper-shaped effort, not a weekend hack.

**Cost.** Hard to bound. Reading the whitepaper carefully + studying
the released weights is ~half-day. Building a credible reproduction is
many days. Validating it works is gated on small-model training runs.

**Status.** Open. Side-quest to the 1-bit-from-scratch direction;
not on the critical path.

**References.**
- `~/projects-new/trx4mr/docs/bonsai_1bit_brief.md` — what's known/proprietary.
- `https://github.com/PrismML-Eng/Bonsai-demo/blob/main/1-bit-bonsai-8b-whitepaper.pdf` — public whitepaper.
- `https://huggingface.co/prism-ml/Bonsai-8B-mlx-1bit` — released weights for inspection.

---

## 1-bit / ternary d6 from scratch — the original Bonsai-inspired pitch (2026-05-07)

**Pitch.** Pretrain a d6 nanochat model with binary `{−s, +s}` (or ternary
`{−s, 0, +s}`) weights from scratch — *not* fp LoRA on a 1-bit base, *not*
post-hoc quantization, but native low-bit pretrain. The strategic prize:
**~14× memory reduction means d12+ might fit on M2** even though pretrain
compute scales the same as fp. The baseline question is "can we even train
a 1-bit d6 to within shouting distance of fp d6's val_bpb 1.174."

**Why interesting / why now.**
- The cosine-NN diagnostic (`docs/cosine_nn_diagnostic_2026-05-06.md`)
  classified d6's failures as ~65 % right-context drift (FP-flavored), and
  the trx4mr Phase 5 fix-direction table prescribes binary/ternary codebook
  coarsening. So this isn't a vibe; it's the diagnostic-supported
  *architectural* fix axis. (Distinct from the lora_proposal's "Bonsai-LoRA"
  arm — see the disambiguation note below.)
- The trx4mr sibling repo already has the infra primitives:
  `picoGPT/binary.py` exposes `BinaryLinear` / `TernaryLinear`, and
  `blabberverse/phase7_arch.py` has a working `--quant {fp,binary,ternary}`
  CLI with per-group-of-128 FP16 scale factors. Porting these into
  `nanochat/gpt.py` as quantized variants of `Linear` is the load-bearing
  infra step.
- The sibling brief `~/projects-new/trx4mr/docs/bonsai_1bit_brief.md`
  (April 2026) lays out Bonsai's binary scheme cleanly: `{−s, +s}`, no zero
  state, FP16 scale per 128-weight group, **no higher-precision escape
  hatches anywhere** (embeddings, attention, MLP, LM head all binary).
  BitNet b1.58 is the *ternary* `{−1, 0, +1}` variant, with which Bonsai
  is sometimes confused — they're distinct designs.

**The Bonsai mystery (caveat).** Bonsai's headline claim is a "native 1-bit
training method" — *not* the standard straight-through estimator. The
training method is proprietary (Caltech research, Hassibi et al., Apache 2.0
weights but closed training code). The published whitepaper at
`https://github.com/PrismML-Eng/Bonsai-demo/blob/main/1-bit-bonsai-8b-whitepaper.pdf`
is the only public source. **Realistically: the operator expects we'll
fall back to STE for our d6 pretrain** (well-trodden, fits our hardware,
matches what trx4mr's `BinaryLinear` already does). Reverse-engineering
Bonsai's actual method is an interesting separate question and may not
fit M2; it's a side-quest, not the load-bearing thing.

**Compute reality check.** 1-bit pretrain is typically *just as compute
heavy* as fp pretrain (or worse, if STE adds overhead). Empirically (2026-05-08
d3 binary trial, ~step 300): **~16 % per-step wall penalty** vs fp32 baseline
on M2. The win is ~14× memory. So: same M2 wall as a fp d6 pretrain (~3 h)
for the d6 validation; then *if* d6 works, d12+ becomes the actual strategic
move because the memory delta is what unblocks larger depth on this hardware.

**⚠ Memory accounting clarification (2026-05-08, see ADR-007).** The "14×
memory → d12 on M2" framing above conflates **training-time** and **inference-time**
memory. STE training as we've ported it (`nanochat/quant.py`) keeps fp32 latents
because the optimizer needs them for gradient accumulation. Training memory
under `apply_quant=binary` is ≥ fp32 training memory (latent + a transient
quantized tensor during forward) — **not 14× less**. The 14× win is realised at
*inference* by serializing weights as sign bits + per-group fp16 scales, which
requires a separate "pack to 1-bit" inference path that we don't have yet.

What this means for this entry's framing:
- "Can we *train* a 1-bit d6 to within shouting distance of fp d6's val_bpb"
  remains the load-bearing Phase 2 question. STE training is what we have;
  it costs ~same compute, ~same memory. The interesting question is the
  quality of the resulting model.
- "d12+ on M2" via this STE port is **inference-only**. d12 *training* on M2
  would still hit the same memory ceiling as fp32 d12. Bonsai's proprietary
  native-1-bit method *might* train with packed weights (separate side-quest);
  we don't.
- Therefore: this entry's strategic prize is "deploy a d12+ binary
  inference checkpoint on M2 *if* we can pretrain it elsewhere or via this
  STE path on enough compute" — which is closer in spirit to the
  lora-proposal-C arm (fp LoRA on a frozen 1-bit base) than originally framed.

**Cost (rough).**
- Phase 1 — port `BinaryLinear`/`TernaryLinear` into `nanochat/gpt.py`,
  swap into `apply_lora`-style `apply_quant` walker, smoke-test forward
  pass + STE backward at d6 scale: ~half-day.
- Phase 2 — d6 binary pretrain (~3 h M2 wall) on canonical recipe; compare
  val_bpb against fp d6 (1.174) and against ternary d6.
- Phase 3 — *if* d6 binary lands within ~5 % of fp val_bpb, the d12+ on M2
  experiment becomes live. Could be many M2-days of experiment work.

**Falsification criteria.**
- d6 binary val_bpb >> 1.5: STE gradient flow fails at d6 scale; the
  operator's "scale up to d12+" pitch needs a different training method
  (potentially Bonsai's mystery method, potentially something else).
- d6 binary val_bpb ≈ 1.18-1.30: STE works; the d12+ memory-unblock thesis
  is testable.
- d6 binary val_bpb ≈ 1.17: STE works *well*; the depth-rule generalises
  cleanly to binary; we're in the interesting regime where 1-bit training
  is a viable design decision for this codebase.

**Status.** Open. **The operator's most-wanted direction.** Currently
gated only on operator priority — infra plan is sketched but not started.

**References / pointers.**
- `~/projects-new/trx4mr/docs/bonsai_1bit_brief.md` — operator's notes on
  Bonsai 1-bit, what's known, what's proprietary, suggested experiments.
- `~/projects-new/trx4mr/blabberverse/phase7_arch.py` — working
  binary/ternary training arch with `--quant` CLI, per-128-group scales.
- `~/projects-new/trx4mr/picoGPT/binary.py` — `BinaryLinear` /
  `TernaryLinear` primitives (the port targets).
- BitNet b1.58: Microsoft, 2024, *The Era of 1-bit LLMs* (the ternary
  prior-art).
- Bonsai whitepaper (linked above) for what's public on the binary scheme.

---

## Local Bonsai 4B/8B as helper models — synthesis, judge, distillation (2026-05-07)

**Pitch.** The operator has Bonsai 4B and 8B running locally; ternary-quantized
inference is fast enough on M2 to use them as **utility models** in our
nanochat workflow:
- **Synthetic training-data generation** for SFT or LoRA experiments
  (current chit-chat / persona curators use Claude CLI which costs the
  subscription; Bonsai-as-generator is free per token).
- **LLM-as-judge** for chat-quality evals where a rubric is too coarse and
  hand-grading is too slow.
- **Distillation target / teacher** for d6 student training — particularly
  interesting in combination with the 1-bit-from-scratch direction
  above, since the teacher's representations may transfer differently
  into a discrete student than into a fp student.

**Why interesting.** This is *infrastructure* in the operator-tooling
sense — it doesn't move the model-quality needle directly, but it makes
several other directions cheaper. The current Claude-CLI curator pattern
costs $3-4 per dataset; a Bonsai 8B local curator costs ~minutes of M2
time with no per-call charge. For experiments where we want 10× the data
or 10× the iterations, the cost crossover is real.

**Cost (rough).** Wiring Bonsai into the nanochat dev pipeline is a
half-day of glue: process invocation + JSON-schema-equivalent prompt
discipline + cost/time bookkeeping. The Bonsai inference path itself
already works locally per the operator (presumably via the
PrismML llama.cpp fork or the MLX 1-bit weights).

**Status.** Open. Independent of the 1-bit-from-scratch direction.

**Risk.** Bonsai's behavioural fingerprint may differ from Claude's
in ways that affect dataset quality. A curator-output diff against the
existing Claude-CLI curators on a small sample (50 conversations) would
de-risk this before committing to it as the default.

---

## ⚠ Disambiguation: three "Bonsai" directions, not one (2026-05-07)

The lora_proposal (`docs/lora_proposal_2026-05-06.md`) and downstream
writeups conflated three distinct experiments under the "Bonsai-LoRA"
heading. They're separate research questions and shouldn't be
collapsed into one work-stream.

| direction | research question | status |
|---|---|---|
| **A. 1-bit/ternary d6 from scratch** (this backlog, above) | Can we *train* a binary base from scratch at d6, with the prize being d12+ memory-unblock on M2? | Operator's original vision; not started. |
| **B. Local Bonsai 4B/8B as utility models** (this backlog, above) | Can we use *their* pretrained binary model to generate data, judge, or teach? | Operator capability; not started. |
| **C. fp LoRA on a frozen 1-bit base** (`lora_proposal_2026-05-06.md` § L1-bonsai) | Does fp LoRA on a *frozen* 1-bit base outperform fp LoRA on fp base for the persona-retention task? | Diagnostic-supported; infra-blocked. |

A and C are easy to confuse. C uses someone else's 1-bit checkpoint as a
*foundation*; A trains our own 1-bit base. Different questions, different
infra, different risks. **The operator's stated vision was A**; C
emerged from the LoRA-proposal frame and the cosine-NN diagnostic.

Both A and C remain interesting; both are open. Keep them distinct in
all future writeups.

---

## Model-growing: d6 → d8 via bert2BERT-style replication (2026-05-07)

**Pitch.** Instead of pretraining d8 from scratch (~17-20h on M2), bootstrap
d8's init from the existing d6 checkpoint:
- Replicate d6's 6 transformer blocks as the bottom 6 of d8.
- Add a width-expansion linear-map for `model_dim` 384 → 512 (nanochat's depth
  rule changes both `n_layer` and `model_dim`, so this is bert2BERT regime, not
  pure layer-stacking — see `bert2BERT`, Chen et al. 2022).
- Zero-init or random-init the top 2 layers.
- Continue pretraining for a fraction of d8's full Chinchilla horizon.

**Why interesting.** No one in this repo has tried it. Reported wins in the
literature are 30-70 % wall-time reduction vs from-scratch to reach a target
loss. On M2 that's the difference between an overnight d8 and a 4-6h d8.

**Tension with project philosophy.** nanochat's design rule is "every depth
produces a compute-optimal model from scratch" (`CLAUDE.md`). A d8-from-d6
init is *by construction* off the compute-optimal frontier — it carries
d6-quality features into a d8-shape. That's fine for "burn fewer M2 hours to
get *a* d8 to play with"; it's not fine if the goal is a clean depth-sweep
data point.

**Cost (rough).** ~20-30 LOC for the width-expansion map + layer replication
helper, plus a one-shot `scripts/grow_d6_to_d8.py`. Continuing pretrain:
~5-10h M2 wall (depending on what fraction of horizon we run).

**Status.** Open. Not blocked on anything. Cheaper than B (LiGO) by far.

**References.**
- Chen et al. 2022, *bert2BERT: Towards Reusable Pretrained Language Models*
  (width + depth expansion in one shot).
- Gong et al. 2019, *Efficient Training of BERT by Progressively Stacking*
  (StackBERT — pure layer-stacking, simpler precedent).
- Chen et al. 2015, *Net2Net* (the seminal function-preserving transforms).

---

## LiGO: learn the small→large weight map (2026-05-07)

**Pitch.** Wang et al. 2023, *Learning to Grow Pretrained Models for
Efficient Transformer Training* — instead of hand-designing the d6→d8 init
map (the bert2BERT direction above), *learn* a linear operator that maps
small-model weights into the larger model's parameter space. State-of-the-art
on this axis at publication.

**Why interesting.** Same motivation as bert2BERT (skip the cost of
from-scratch pretrain at the larger scale) but with a learned mapping instead
of a hand-crafted one. Empirically beats bert2BERT and stacking baselines in
the paper.

**Cost (rough).** Materially more code than bert2BERT — needs the LiGO
linear-operator parameterisation, a meta-training loop on the operator, and
then the actual continued-pretrain at d8. A reasonable order-of-magnitude is
"a week of focused work" rather than "a weekend hack". Ranks behind A
(bert2BERT) on cost-to-information ratio for nanochat-scale.

**Status.** Open. Probably only worth it if the bert2BERT-style direction
shows enough signal to want to push further on weight-mapping quality.

**References.**
- Wang et al. 2023, *Learning to Grow Pretrained Models for Efficient
  Transformer Training* (ICLR'23, "LiGO").
