# Hope/NL phase audit — 2026-05-03

A clean cross-stage view: what the original ticket planned, what's been
done, what's left, what each option would cost. Synthesis of HANDOFF.md
and the per-stage writeups.

Source of the staged plan: `~/projects-new/trx4mr/docs/idea-hope-nested-learning.md`
(§"Staged implementation path", lines 102–153).

## Stage matrix

| # | Plan | Status | Result | Writeup |
|---|---|---|---|---|
| 0 | `memory_state` plumbing through `forward()`, return `None` initially | ✅ shipped | bit-identical loss/logits with vs without; 6 contract tests | `docs/hope_nl_stage0_2026-05-01.md` |
| 1 | Swap MLP at one block for fast-weight memory; **fixed** α, η; per-seq reset | ✅ shipped | val_bpb 1.179 vs baseline 1.174 (+0.4%); SFT 0.6712 vs 0.6639 (+1.1%) | `docs/hope_nl_stage1_2026-05-01.md`, `docs/hope_nl_stage1_full_pretrain_2026-05-01.md` |
| 1.5 | (unplanned, Codex-driven) MQAR synthetic probe | ✅ shipped | Found `W_o=0` cold-start trap; `hope_memory_w_o_init_scale=1.0` is the fix; additive vs swap topology characterized | `docs/hope_nl_stage1_5_*` (5 files); ADR-001 |
| 2 | Per-token learned `α_t`, `η_t` via `sigmoid` projections | ✅ shipped | Pretrain val_bpb **1.1743** (parity); SFT val_bpb **0.6518** ⭐ (–1.8% vs baseline, –2.9% vs Stage 1 swap); ChatCORE in flight | `docs/hope_nl_stage2_2026-05-02.md`; ADR-002 |
| 3 | Chunk-parallel dual form per the paper | ⏸ not started | — | — |
| 4 | Multi-block memory (swap FFN in *some* layers, not just one) | ⏸ not started | — | — |
| 5 | CMS — multi-frequency memory branches (fast/mid/slow) | ⏸ not started | — | — |
| 6 | Full Hope: self-modifying memory modules | ⏸ stretch | — | — |

## Where we landed at the original-ticket level

The original ticket asked: "does this architecture work, and is it
worth the trouble at the project's scale?"

**Answer at d6:** It works, runs stable, matches baseline on val_bpb,
and **wins on SFT val_bpb** by a small but real margin. The probe-level
yellow flags from session 3 (bimodal optimization, seed sensitivity at
T=128) **did not manifest** at full pretrain horizon (5000 iters,
T=512). Single-seed result, but separation is well above the d6 noise
floor.

The honest read: Hope/NL's dramatic benefits (long-context recall,
test-time adaptation) live at scales we won't reach on M2. What we have
is empirical evidence that the architecture is competitive at d6 and
training is well-behaved.

## Open items (not stage-numbered)

These came up during sessions 2–4 and are punted for future work:

| Item | Cost | Why it matters | Source |
|---|---|---|---|
| **ChatCORE on Stage 2 SFT** | ~2h (in flight now, run `2df5c88o`-equivalent) | First downstream-task signal on Stage 2 | HANDOFF.md:129 |
| **Multi-seed Stage 2 pretrain** | ~3h × N seeds | Confirms val_bpb 1.1743 isn't single-seed luck | HANDOFF.md:541 |
| **Stage 2 swap-topology pretrain** | ~3h | Probe found swap and additive equivalent at synthetic level; LM val_bpb gap unknown | HANDOFF.md:540 |
| **Stage 1-additive (W_o=1) pretrain** | ~3h | MQAR found parity with baseline; whether DCLM val_bpb agrees was punted | HANDOFF.md:541 |
| **Hope-specific behavioral probe** | ~half day to design + run | MQAR was the synthetic; in-context binding / parity / counting might surface different tradeoffs | HANDOFF.md:327 |
| **Lift always-final-layer-L constraint in `_compute_window_sizes`** | ~1h patch + probe re-run | Would let MQAR run with restricted attention to genuinely test memory-only recall | HANDOFF.md:326 |
| **`LearnedGateLinearMemory` forward-pass fuse** | ~half day | Combine `log_decay→decay` into one in-place op; modest M2 win | HANDOFF.md:543 |

## Recommended next moves (operator decision)

After Codex sync (2026-05-03), refined sequence:

**A1. ChatCORE on Stage 2 SFT** ✅ *done 2026-05-03, ChatCORE = 0.1744*
- First downstream-task signal on Stage 2. SpellingBee 95.31% dominates 91% of the metric; non-SpellingBee tasks all at noise floor. Writeup at `docs/hope_nl_stage2_chatcore_2026-05-03.md`.

**A2. SFT-seed-variance disambiguation** ✅ *done 2026-05-04*
- Three SFT seeds (42, 1, 2) on the same Stage 2 pretrain → val_bpb 0.6518 / 0.6516 / 0.6520. Spread 0.0004, ~30× smaller than the 0.0121 headline win.
- **Conclusion: SFT is highly seed-stable; headline is not an SFT-seed lottery. Variance, if any, lives in pretrain → A3 justified.**
- Writeup: `docs/hope_nl_stage2_seed_variance_2026-05-04.md`.

**A3. Multi-seed Stage 2 pretrain** *(~6h, queued for next session)*
- A2 confirmed SFT-seed-stability → A3 is the right path. Two more pretrains with --seed=1 and --seed=2 on the existing d6 Stage 2 config. Compare val_bpb at step 5000 against the seed=42 reference (1.1743).
- n=3 total is **directional** ("happened in 1/3, 2/3, 3/3"), not a confidence interval. Frame accordingly.
- Metadata audit done: seed plumbing landed (commit `29146e7`), captured in meta_*.json automatically. --sft-tag separate save dir (commit `fc48d9c`). Recipe lifted to key_facts.md (commit `f6467ff`). **--inherit-from for guaranteed config parity (commit `ca9bc94`)** loads reference user_config as parser defaults; A3 launch commands are now ~5 lines instead of ~20, with parity by construction.
- Launch commands captured in the A2 writeup.

**Claim framing — important** (per Codex sanity-check 2026-05-04, Q1=a):

A3 will support: **"Stage 2 pretraining is seed-stable and reaches baseline-like pretrain bpb."**

A3 will **NOT** independently support: "Stage 2's architecture effect is real vs natural d6-seed noise."

A2 (multi-seed SFT, 3/3 within 0.0004) only bounds the SFT-seed component of variance. It says nothing about pretrain-seed noise on the *baseline* d6 architecture, which we never measured (and the baseline checkpoint is gone). Without that, multi-seed Stage 2 pretrain alone establishes internal consistency, not architectural significance.

**A3-prime** *(cheaper bound on the baseline-staleness risk; ~4h)*
- After A3 if results are tight: one fresh baseline d6 pretrain + SFT with the modern recipe + config audit.
- Codex's compromise: doesn't fully bound multi-seed baseline variance (would need n=3 baseline = ~13h), but catches the biggest specific risk — that the historical val_bpb 1.174 baseline is a stale/lucky/unlucky artifact that wouldn't be reproduced by the current optimizer/recipe.
- A3-prime turns "Stage 2 beats baseline by 1.8% on SFT" into "Stage 2 beats *modern-recipe* baseline by X% on SFT" — same comparison, post-recipe-drift. Useful regardless of whether we want full multi-seed baseline.

**B. Stage 4 (multi-block memory)** *(half day design + ~3h × {2,3} configs)*
- Conditional on A3 surviving. First stage where memory is doing structural work, not a single-layer accent.
- 2–3 configs: e.g., layers {2,3,4} vs all-except-first/last vs all.
- Risk: optimizer pressure scales with how many layers carry memory; bimodal basin behavior we saw at probe scale could resurface.

**C. Wrap and write up** *(2–4h synthesis)*
- Default if A2 falsifies the win or budget runs out.
- Stage 0–2 is a clean experimental unit either way; honest answer to the original ticket is documentable today.

## My read

Sequence: **A1 → A2 → (A3 → B) | C**.

A2 is the cheapest experiment that could change the headline. Worth
~2.5h before committing 6h of pretrain budget on a finding we can't
yet localize between pretrain and SFT.

If A2 confirms SFT-seed-stability, A3 → B is the right path: stage 0–2
have all been incremental architectural additions, and Stage 4 is where
memory starts doing structural work.

If A2 falsifies the SFT win, C is the honest stop. The trx4mr ticket
gets answered with "architecture trains stable at d6, downstream
benefits did not survive seed variance, real benefits live at larger
scale" — itself a clear research contribution.

## Status checklist (mirrored from HANDOFF.md)

For convenience — the unchecked boxes:

- [x] Full Hope/NL paper §4–§9 obtained ← **resolved 2026-05-03** (read NL.pdf, 40pp). Web sources (learnopencv, grokipedia) 403'd, but unnecessary now.
- [x] ChatCORE on final Stage 2 SFT checkpoint — **0.1744** (2026-05-03), SpellingBee dominates 91%; writeup at `docs/hope_nl_stage2_chatcore_2026-05-03.md`
- [ ] Stage 1-additive (W_o=1) full pretrain
- [ ] Stage 2 swap-topology full pretrain
- [ ] Multi-seed confirmation of Stage 2 val_bpb 1.1743

## Source bootstrap from full paper (2026-05-03)

The full NeurIPS 2025 version of Behrouz et al. unblocks Stages 3/5/6.
Capturing what's now known:

### CMS — Continuum Memory System (§7)

Forward (Eq 70): chain of MLP blocks at different update frequencies:
```
y_t = MLP^(f_k)(MLP^(f_{k-1})(... MLP^(f_1)(x_t)))
```

Update (Eq 71): each level updates only every C^(l) steps with gradient sum:
```
θ^(f_l)_{i+1} = θ^(f_l)_i - Σ_{t=i-C^(l)}^i η^(l)_t · f(θ^(f_l)_t; x_t)   if i ≡ 0 (mod C^(l))
```
Where `C^(l) := max_i C^(i) / f_l` (no ceil in the source). Self-referential
as written; **for implementation, treat `max_i C^(i)` as a configured base
chunk size and derive per-level frequencies from it** (per Codex sanity-check
2026-05-04). Eq 71 is **formally** sparse-in-token-time (only update at
chunk boundaries); **practically** implemented via chunk-parallel training
that processes all tokens in a chunk in parallel and applies the boundary
update once.

Three CMS variants in the paper:
- **Nested** (Eq 72): level s+1 init meta-learned in level s
- **Sequential** (Eq 73): output of s feeds into s+1, backprop through chain — **the paper's main CMS / Hope path**
- **Independent (head-wise)** (Eq 74): parallel blocks combined via `Agg(MLP^(f_k)(x_t), ..., MLP^(f_1)(x_t))` — explicitly defined; structurally simpler and decoupled, but **not the topology Table 6's "w/o CMS" ablation was measured against**

### Hope architecture (§8.3, Eq 94-97)

```
o_t = M_memory,t-1(q_t)                                         (self-mod Titans output)
v̂_□,t = M_□,t-1(v_t)                                            (each memory generates own values)
M_□,t = M_□,t-1(α_t I - η_t k_t k_t^T) - η_t ∇L                  (DGD with weight decay)
y_t = MLP^(f_k)(MLP^(f_{k-1})(... MLP^(f_1)(o_t)))               (CMS chain on top)
```

Plus L2 normalization on q,k, plus local convolutions of window 4. The
formal equations omit these "for clarity"; experiments use them. Nanochat
already has QK norm in its attention path, but Stage 1/2 memory modules
do **not** normalize their internal memory q/k.

**Hope-Attention variant** (paragraph after Eq 97): replace self-modifying
Titans (Eq 94-96) with softmax global attention. The faithful version is
**attention output → sequential CMS chain**, which structurally **replaces /
adapts the post-attention MLP path** (in nanochat: the existing ReLU² MLP
at each block). Not "add a third residual branch" — that would be additive
memory, which is what Stage 1-additive / Stage 2 already does. Per Codex
sanity-check 2026-05-04: budget option E as a real design block, not a
trivial CMS append. Local conv-4 is plausibly not load-bearing for the
Hope-Attention variant; safer to start without it but acknowledge a no-conv
version isn't paper-exact.

### M3 — Multi-scale Momentum Muon optimizer (§7.2, Algorithm 1)

```
M^(1)_t = M^(1)_{t-1} + β_1 · g_t                  (every step, fast first momentum)
M^(2)_t = M^(2)_{t-1} + β_3 · Σ g_i (chunk sum)     (every f steps, slow memory)
V_t     = V_{t-1} + β_2 · g_t²                      (every step, AdamW variance)
O^(1)_t ← NewtonSchulz_T(M^(1)_t)
O^(2)_t ← NewtonSchulz_T(M^(2)_t)                   (every f steps)
Θ_t ← Θ_{t-1} - η · (O^(1)_t + α · O^(2)_t) / (√V_t + ε)
```

M3 = Adam (V_t variance) + Muon (NewtonSchulz on both momenta) + CMS
(M^(2) chunk-summed) — **shorthand only**. Algorithm 1 is **not**
production AdamW/Muon: no bias correction on V_t, no decoupled weight
decay (per Codex sanity-check 2026-05-04). Paper Figure 12 confirms M3 is
slower than Muon at 140M and 1.3B; explicitly marked proof-of-concept.
**Ignore for nanochat unless optimizer research becomes the project goal.**

### Chunk-parallel training (§8.2, Eq 90-93)

Split input into ⌈L/C⌉ chunks of C tokens. Update memory once per
chunk using gradient accumulated across the chunk. Same approach as
Sun 2024 (TTT) and Behrouz 2025c. Solves Stage 3 of our ticket.

For our Stage 1/2 module (linear attention memory), we'd:
1. Process chunk in parallel (no Python loop within chunk).
2. Update `M_t` based on chunk-final state.
3. Take gradient w.r.t. previous chunk's final state for the next step.

### §7.3 Ad-hoc level stacking — direct retrofit recipe

> "Initialize the parameters in a level with a model's pre-trained
> weights. More formally, given a CMS with `{MLP^(f_l)}`... we use the
> trained parameters of `{MLP_pretrained_l}` as the initial state of
> CMS blocks."

In the paper's experiments they retrofit Llama-3-8B this way and continue
pretraining for 15B tokens. **Mechanically applicable** to our Stage 2 d6
pretrain checkpoint, but **not scientifically proportional** at d6 / ~80M
token budget (per Codex sanity-check 2026-05-04). The recipe depends on
useful pretrained MLP weights AND enough continued-pretrain tokens for the
lower-frequency levels to update meaningfully. At our budget, low-freq
levels may barely update. Treat as exploratory stress test, not a
proportional mini version of their result.

### Ablation evidence on what each component buys (Table 6)

| Removed component | Wiki ppl change | Reasoning acc change |
|---|---:|---:|
| (full Hope baseline) | 12.24 | 58.1 |
| w/o DGD | +9.6% | -1.6 |
| w/o Momentum | +11% | -1.2 |
| w/o weight decay | +12% | -0.9 |
| **w/o CMS** | **+6.5%** | **-0.8** |
| w/o inner-q | -0.4% (noise) | -0.7 |
| w/o inner-v | +13.5% | -3.0 |

**Important wording fix** (per Codex sanity-check 2026-05-04): the
"inner-projection k/v/q" ablations in Table 6 are projections **moved
from higher-frequency to lowest-frequency level**, not deleted. Inner-q
near-neutral means **adaptive inner query projection is unnecessary** in
their setup; it does **NOT** mean a regular query projection is removable.
Inner-v is the worst ablation (+13.5%) but inner-k is also important
(+13.77%). Practical takeaway for our future Stage 5/6 implementations:
likely safe to skip the *adaptive-q* mechanism, but keep adaptive-k
and adaptive-v (and standard q,k,v projections obviously stay).

**Honest scale caveat**: paper's results are at 760M/30B and 1.3B/100B
(~600× our d6). The ~6.5% perplexity from CMS alone might be
proportionally smaller (or larger, or zero) at d6. We don't know.

## Updated stage map with what we now know

| # | Original plan | Now-known equations | Cheapest experiment |
|---|---|---|---|
| 3 | Chunk-parallel | Eq 90-93 (§8.2); references Sun 2024 / Behrouz 2025c | Refactor `LearnedGateLinearMemory` for chunk-parallel form. Allows T > 512 within current memory budget. |
| 4 | Multi-block memory | Naturally aligned with what Hope does (§8.3, Fig 5: stack of CMS blocks) | Re-run Stage 2 with `hope_additive_memory_layer` at multiple layers (config flag already exists; small code change to accept a list). |
| 5 | CMS multi-frequency | Eq 70-71, Eq 74 (independent variant simplest) | Add CMS-Independent on top of Stage 2's output: parallel MLPs at frequency 1, ½, ¼, ⅛ combined via learned weighted sum. |
| 6 | Self-modifying memory | Eq 83-90 (full self-mod Titans) | Probably skip — too many moving parts at d6. Paper's full Hope wins by stacking many things; w/o CMS it's still a worthwhile architecture. |

## New options to consider

In addition to the A1/A2/A3/B/C from above, the paper bootstrap unlocks:

**D. §7.3 retrofit experiment** *(exploratory stress test; ~3-6h)*
- Take the Stage 2 d6 pretrain checkpoint. Place the existing MLP blocks at different CMS levels via Eq 71 update schedule. Continue pretraining for a reduced token budget.
- **What we'd learn**: whether the retrofit mechanism even fires at d6 scale. Per Codex 2026-05-04, this is **not a proportional mini version** of paper's result — at our token budget, lower-frequency levels may barely update.
- Risk: chunk-update logic in optimizer adds ~half day of implementation work; result might be uninformative (level barely moved from init = "no signal" not "no benefit").

**E. Hope-Attention as Stage 5 target** *(real design block; ~1-1.5 weeks)*
- Replace nanochat's post-attention ReLU² MLP at one or more blocks with a sequential CMS chain (Eq 70 + Eq 71 update schedule). Keep nanochat's existing attention (already has QK norm). Per Codex 2026-05-04: **structural change, not "just append CMS"** — refactors the post-attention path itself.
- Keep nanochat's native ReLU² MLP form per level to avoid SwiGLU-vs-ReLU² confound.
- **Pass/fail bar**: probe via `dev/probe_mqar.py` first. If saturation step ≥ baseline, proceed to pretrain.
- More ambitious than Stage 4 (multi-block memory). Closer to a real Hope implementation, sidestepping the highest-risk component (self-modifying Titans).

**F. CMS-Independent ablation isolated** *(cheap CMS signal; ~3h)*
- Add Eq 74 head-wise CMS to our existing model with no other changes. Parallel MLP chain at varying chunk sizes, combined via learned weighted sum.
- **What we'd learn**: a *cheap signal* on whether multi-frequency memory at d6 produces any val_bpb effect. **NOT** directly comparable to Table 6's "w/o CMS" — that ablation was on Sequential CMS within full Hope; our F isolates Independent CMS on a vanilla d6 baseline (per Codex 2026-05-04). A positive signal motivates E (faithful Sequential implementation); a null signal is informative but not falsifying for the paper's claim.

## Refined recommendation order

1. **A1**: ChatCORE on Stage 2 SFT ✅ done 2026-05-03
2. **A2**: SFT-seed-variance disambiguation ✅ done 2026-05-04
3. **A3**: multi-seed Stage 2 pretrain — **queued for next session, ~6h**
4. **A3-prime** (conditional on A3 tight): one fresh baseline d6 pretrain + SFT with modern recipe (~4h) — bounds the historical-baseline-staleness risk without paying for full multi-seed baseline
5. **F**: CMS-Independent ablation as **cheap CMS signal** at d6 (~3h) — not a Table 6 analogue; tells us if multi-frequency memory has *any* val_bpb effect at our scale
6. **D** OR **E**: retrofit (exploratory) OR Hope-Attention (real design block, ~1-1.5 weeks)
7. **C**: wrap and write up

The paper bootstrap pulls the locus of remaining work toward CMS,
which the paper's ablation says contributes ~6.5% of Hope's gain
at 760M-1.3B scale (caveat: ours is ~600× smaller, transfer unknown).
F is the cheapest way to find *any* d6-scale signal on the multi-
frequency mechanism; if positive, motivates the bigger E investment.
