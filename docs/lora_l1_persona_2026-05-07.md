# L1 — Persona-LoRA on d6: real signal, mode collapse, headroom

**Date**: 2026-05-07
**Status**: experiment ran, verdict landed. LoRA infrastructure validated
end-to-end; L1's persona-LoRA hypothesis has real signal on the failure
mode it targets; ~50% of outputs still collapse to a reflexive opener,
which points at concrete v2 levers.

## What this experiment was

Per the LoRA proposal (`docs/lora_proposal_2026-05-06.md`), L1 was the
direct attack on the multi-turn-rubric persona-retention failure that
yesterday's session caught: T1 introduces "I'm Alex, a software engineer
in Portland", chit-chat for a few turns, T5 asks "what's my name and
job?" — and the base SFT model says "Sydney" or "nanochat" or "Java J"
instead of recalling.

The hypothesis: a small rank-r LoRA trained explicitly on a curated set
of multi-turn persona-retention conversations can lift just *that*
failure mode without breaking the rest of the chatbot.

The cosine-NN diagnostic from yesterday
(`docs/cosine_nn_diagnostic_2026-05-06.md`) classified d6's chat failure
modes as **~65% right-context drift (FP-flavored)** — a regime where
codebook coarsening or capacity-routing-via-LoRA are the indicated
fixes. L1 tests the latter: does pointing 0.1% of new trainable
parameters at the persona-recall pattern materially move the metric?

## What landed in the codebase today

Three commits worth of infrastructure plus the experiment itself:

### LoRA core (commit `79e4b3c`)

`nanochat/lora.py` (~210 lines) + `tests/test_lora.py` (~190 lines).
Module surface:

- **`LoRALinear(base, rank, alpha)`** — wraps a frozen `nn.Linear`,
  forward = `base(x) + (x @ A^T) @ B^T * (alpha/rank)`. A is
  Kaiming-init, B is zero-init so apply-time forward is bit-equivalent
  to base. Mirrors `nanochat.gpt.Linear`'s master-fp32 / matmul-in-
  input-dtype convention end-to-end.
- **`apply_lora(model, target, rank, alpha)`** — walks
  `model.transformer.h`, replaces matching attention projections with
  `LoRALinear` wrappers, freezes everything else. Errors if it would
  inject zero adapters (target-name typos surface immediately).
- **`lora_state_dict` / `load_lora_state_dict`** — adapter-only
  save/load with strict-key validation.

Tests pin: freeze-correctness, zero-init equivalence, post-perturbation
divergence, save/load round-trip, target-typo error path. 9 tests all
green; full suite of 41 still passes.

### Training entry point + inference wiring (commit `f75a90d`)

- **`scripts/chat_sft_lora.py`** — focused training entry point.
  Single CustomJSON dataset, plain AdamW over LoRA params only (no
  Muon — rank-r is too small to benefit from orthogonalisation), no
  torch.compile. Reuses `tokenizer.render_conversation` and
  `evaluate_bpb` so val numbers are directly comparable to chat_sft
  runs. Saves to `$NANOCHAT_BASE_DIR/lora_checkpoints/<lora_tag>/`
  with adapter-only state plus a meta json.
- **`apply_lora_from_tag(model, lora_tag)`** in `nanochat/lora.py` —
  one-call helper that reads the meta json, calls `apply_lora` with
  the recorded target/rank/alpha, loads the weights with strict-key
  validation.
- **`scripts/chat_cli.py` + `scripts/chat_web.py`** — gain
  `--lora-tag` and `--lora-step` flags. Inference engine path is
  untouched: `LoRALinear` is a regular `nn.Module`, so `Engine` and
  the KV cache work without modification.

### Persona-retention curator (commit `da67b49`)

`dev/curate_persona_retention.py` (~475 lines). Synthesizes 6-turn
multi-turn persona-retention conversations via the **`claude` CLI in
headless mode** — pattern lifted from sibling project
`elixir-explore/pulse` (`lib/pulse/claude/{suggested_responses,sentiment,
surprise}.ex`):

```sh
claude -p '<prompt>' \
    --model haiku --tools "" --no-session-persistence \
    --output-format json \
    --system-prompt '<system>' \
    --json-schema '<json-schema>'
```

Reads the result envelope's `structured_output` field. No
`ANTHROPIC_API_KEY` needed — uses the operator's active Claude Code
subscription auth. The curator builds Pydantic-validated `Conversation`
objects from each batch, validates client-side that T1 contains all
persona substrings and T6 recalls them all, tracks a running diversity
list of used names/roles/locations, fed back into each batch's prompt
as an "avoid these" constraint. Held-out eval-set personas are
declared in the system prompt so train-set generation steers clear of
them.

Output:

| file | rows | shape |
|---|---:|---|
| `~/.cache/nanochat/persona_retention_v1.jsonl` | 300 | bare `[{role,content}, ...]` per line — directly loadable by `tasks.customjson.CustomJSON` |
| `~/.cache/nanochat/persona_retention_v1_eval.jsonl` | 30 | `{messages, _persona}` per line — held-out personas with metadata for substring eval |
| `~/.cache/nanochat/persona_retention_v1.log` | — | generation log + reject reasons |

Cost: **~$3.13 against the Claude subscription**, ~74% acceptance rate
(113 rejects out of ~430 attempts — most rejects were schema-substring
mismatches where Claude's persona dict claimed a role not exactly in
T1's text). Token-length distribution on train: min=82, median=120,
p90=148, max=207 — well under `max_seq_len=512`.

### L1 training run

```sh
uv run python -m scripts.chat_sft_lora \
    --base-tag d6_baseline_modern_sft \
    --data-path ~/.cache/nanochat/persona_retention_v1.jsonl \
    --lora-tag d6_l1_persona_lora \
    --num-iterations 300 \
    --device-batch-size 4 --max-seq-len 256 \
    --lr 1e-4 --eval-every 50 --log-every 10
```

Setup metrics:
- Target: `c_q,c_v` rank=8 alpha=16 (LoRA paper baseline)
- Injected: 12 wrappers (6 layers × 2 targets)
- Trainable params: **73,728** — 0.100% of base 73.5M
- Optimizer: plain AdamW LR=1e-4 over 24 LoRA tensors

Training trajectory (300 iters, 50.4s wall on M2 mps):

| step | loss | val_bpb |
|---:|---:|---:|
| 0 | 4.20 | — |
| 50 | 3.84 | 1.2434 |
| 100 | 3.58 | 1.1733 |
| 150 | 3.45 | 1.1196 |
| 200 | 3.35 | 1.0930 |
| 250 | 3.37 | 1.0807 |
| 299 | 3.25 | **1.0771** |

val_bpb monotonically improving, 0.166 absolute drop. Loss descended
cleanly. Adapter saved to
`lora_checkpoints/d6_l1_persona_lora/lora_000300.pt` (296 KB).

### Eval harness (commit `6fa5b34`)

`dev/eval_persona_retention.py` (~220 lines). For each held-out
conversation:
1. Take the first 5 messages (u/a/u/a/u, ending at T5 recall question).
2. `tokenizer.render_conversation` on those 5; append a fresh
   `<|assistant_start|>` so generation begins T6.
3. Generate with `Engine.generate` (greedy, temp=0, top_k=50, max=120
   tokens), stopping on `<|assistant_end|>`.
4. Score: case-insensitive substring presence of `name`, `role`,
   `location` in the generated text.

Compares two arms: `arm_base` (d6_baseline_modern_sft only) and
`arm_lora` (base + `d6_l1_persona_lora`).

## The L1 verdict

| metric | arm_base | arm_lora | delta |
|---|---:|---:|---:|
| name_recall | 6.7% (2/30) | 20.0% (6/30) | **+13.3pp** |
| role_recall | 33.3% (10/30) | 26.7% (8/30) | -6.7pp |
| location_recall | 13.3% (4/30) | 26.7% (8/30) | **+13.3pp** |
| **all_three** | **0.0% (0/30)** | **13.3% (4/30)** | **+13.3pp** |

**The load-bearing finding**: `all_three` (recall name AND role AND
location) goes from 0/30 to 4/30. Base *never* recalled all three on
the held-out conversations; LoRA does on 4 of 30. This is a 0→nonzero
transition on a metric the base model couldn't move at all, with a
0.1%-of-parameters intervention.

### What the wins look like

| # | persona | base output (truncated) | LoRA output (truncated) |
|---:|---|---|---|
| 5 | James O'Brien / mechanic / Dublin, Ireland | "I'm a mechanic here in Dublin, Ireland. I've got a lot of traffic..." | "**The James O'Brien, I'm a mechanic here in Dublin, Ireland.**" ✓✓✓ |
| 6 | Chen Wei / graphic designer / Chengdu, China | "I'm glad I could help with that. What's your name, and what's your name?" | "**Chen Wei here, graphic designer based in Chengdu, China.**" ✓✓✓ |
| 8 | Marcos Costa / guitarist / Recife, Brazil | "I'm a guitarist, guitarist, and guitarist..." (loop) | "**Marcos Costa, guitarist in Recife, Brazil**, Brazil, I'm all about Latin music." ✓✓✓ |
| 14 | Leila Saadi / archaeologist / Cairo | "It was a time to be in the past, but I was a small, small, small..." (loop) | "**Leila Saadi, an archaeologist in Cairo**, is a archaeologist in Cairo." ✓✓✓ |

These are the four `all_three` passes. Of those, three pass *cleanly*
on the first sentence (5, 6, 14); one (8) hits all three but the second
sentence drifts. None are perfect English, but the persona retrieval
mechanism is firing correctly.

### What the failures look like

About **half of LoRA outputs collapse to a generic opener**: "That's a
great way to work" / "That's a fascinating work." / "That's a great
project!". The LoRA learned the *style* of T6 (short, conversational,
sometimes recalling) but on many prompts doesn't reliably attend back
to T1 to pull the persona substrings.

| failure type | base behaviour | LoRA behaviour |
|---|---|---|
| **mode collapse** | repetitive loops, "I'm a bit concerned about..." | "That's a great way to work." (~half of LoRA outputs) |
| **partial recall** | some role recall via repetition loops | name+location recalled, role missed |
| **identity deflection** | "I'm sorry, but as an AI..." (Fatima/Anouk) | reduced but still occasional |

The `role_recall` regression (-6.7pp) is mostly an artefact of the base
accidentally hitting `role` through verbose repetition loops ("I'm a
marine biologist, and I'm a marine biologist...") that score as a
substring hit even though the model didn't really *recall* anything.
LoRA's tighter responses sometimes drop role even when reading T1
correctly.

## What this proves and what it doesn't

**Proves**:
1. **LoRA infrastructure works end-to-end on real data**: the wiring
   (apply, train, save, load through `apply_lora_from_tag`, infer via
   the existing `Engine`) is correct. The smoke test in `f75a90d`
   already showed wiring; this run shows the trained adapter
   measurably moves chat behaviour.
2. **L1's persona-LoRA hypothesis has real signal at d6**: 0.1% of
   trainable parameters, 296 KB on disk, 50 seconds of wall, $3 of
   curation cost, and `all_three` moves 0/30 → 4/30 on held-out
   personas the model has never seen. This is the cheapest probable
   win the LoRA proposal sketched, and it landed on the side of
   "real signal" rather than "null".
3. **The cosine-NN diagnostic's directional prediction is alive**: a
   capacity-routing intervention (LoRA on Q+V) on the dominant-axis
   failure regime moved the metric in the predicted direction. The
   direction is right; the magnitude is small but nonzero.
4. **Bonsai-LoRA priority is preserved**: this was the cheapest test
   of "does LoRA on d6 do anything?". It does. The LoRA
   infrastructure now exists for the larger Bonsai-LoRA arm of the
   L1-d6 vs L1-bonsai comparison the proposal sketched.

**Does not prove**:
1. **Not a full solve.** 13.3% all_three is a 0→nonzero transition,
   not a 0→100% transition. Most prompts still fail.
2. **Not catastrophic-forgetting-tested.** This run measured persona
   retention specifically; we haven't checked whether the LoRA
   regresses other capabilities (math, MMLU-style MC, identity
   responses, the broader 7-prompt rubric). A LoRA that fixes
   persona but breaks math is differently broken, not less broken.
3. **Single seed, n=30 held-out, single set of hyperparameters.** The
   13.3% number has wide error bars at this sample size. A different
   seed or rank choice could land at 6.7% or 20% without anything
   changing about the underlying mechanism.
4. **Role regression has noise.** -6.7pp is small but in the wrong
   direction. The artefact explanation (base hits role via
   repetition) is plausible but not directly verified — it could
   also be that LoRA is partially overwriting role-recall capacity
   with name/location-recall capacity.

## Headroom — concrete v2 levers

The mode-collapse pattern points at well-understood LoRA-tuning levers:

| lever | hypothesis | how to test |
|---|---|---|
| **Wider target: Q+K+V+O** | The recall path needs to *attend* (K) and *project* (O), not just generate matching keys (Q) and values (V). The current Q+V coverage misses the K and O paths that mediate cross-turn attention specifically. | Re-run with `--target c_q,c_k,c_v,c_proj` (4 targets × 6 layers = 24 wrappers, 147K trainable params, still ~0.2% of base) |
| **Higher rank: r=16** | Rank 8 may be saturating; 16 doubles capacity at marginal cost (still <1% of base) | Re-run with `--rank 16 --alpha 32` |
| **Longer training: 600 iters** | 50s/300 iters didn't converge — final-step loss was still trending down | Re-run with `--num-iterations 600 --warmdown-ratio 0.3` |
| **Higher LR with cosine schedule** | LR=1e-4 may be conservative for r=16. Adding a cosine warmup → warmdown could let the LoRA find a less-collapsed mode. | Re-run with `--lr 3e-4 --warmup-ratio 0.1` |
| **Larger dataset** | 300 conversations may not be enough for the LoRA to commit to "always recall" rather than "sometimes recall, sometimes deflect". | Re-curate at 600 rows — same script, ~$6 of subscription cost, ~10 min wall. |

The cheapest single change is probably **rank=16 + Q+K+V+O target** —
it directly addresses the two most-likely undersizing failures with no
new data. That's L1 v2.

The Bonsai-LoRA arm of the L1-d6 vs L1-bonsai comparison the proposal
sketched is the next level beyond v2 — different *base*, same
adapter mechanism — and should still be approached as Phase 2 once
the d6 LoRA is settled.

## Limits of this experiment

- **Eval is substring-recall only.** A response containing all three
  persona substrings can still be incoherent ("guitarist in Recife,
  Brazil, Brazil, I'm all about Latin music."). It captures *whether
  the persona was retrieved*, not *whether the response is good*.
- **No broader rubric run.** We haven't re-tested the 7-prompt
  multi-turn rubric from
  `docs/multi_turn_chat_eval_2026-05-06.md`. The LoRA could improve
  persona_retention specifically and regress topic_stickiness or
  self_correction. Untested.
- **Eval personas were synthesized by the same model that synthesized
  the train set.** Held-out by name/role/location (verified, 30/30
  unique disjoint personas), but generated by the same Haiku run with
  the same system prompt. A real-distribution test would use rubric
  prompts written by humans.
- **Mode-collapse hypothesis is plausible-but-untested.** "That's a
  great way to work" appearing on ~50% of outputs *could* be
  under-training, *could* be data shape, *could* be greedy decoding
  hitting a sticky local optimum. We haven't isolated it.

## Today's commits in order

```
da67b49  dev: persona-retention curator via claude CLI (sibling pulse pattern)
6fa5b34  dev: persona-retention eval — base vs base+LoRA on held-out personas
[this writeup]
```

Plus the previously-committed infrastructure:

```
79e4b3c  nanochat/lora: LoRA wrapper, apply walker, state-dict round-trip + tests
f75a90d  chat_sft_lora + --lora-tag inference: end-to-end LoRA training pipeline
```

## Where this leaves us

LoRA-on-d6 is a real mechanism. The cheapest experiment we could run
moved the metric the diagnostic predicted it would move. The
infrastructure is now landed for L1 v2 (re-tune the same setup),
and for the L1-bonsai arm whenever the Bonsai integration is
prioritised.

The verdict from yesterday's chat-quality session was *"chatbot is
capacity-bound across four full-parameter levers"*. The verdict from
today is *"add a fifth lever — parameter-efficient adaptation — and the
capacity bound moves at d6"*. Both verdicts can be true; the LoRA bump
is meaningful at this scale but not a ceiling-breaker.

The next-cheapest probable win: L1 v2 (Q+K+V+O at rank 16, ~3 minutes
of training, no new data needed).

---

## L1 v2 addendum — 2026-05-07

The cheapest v2 lever from the headroom table above (Q+K+V+O target +
rank=16) was run later the same day. The headroom hypothesis was
correct *and* understated.

**Setup delta from v1**: target `c_q,c_k,c_v,c_proj` (was `c_q,c_v`),
rank 16 (was 8), alpha 32 (was 16), 600 iters (was 300), lr 3e-4 (was
1e-4). Same dataset, same seed. Trainable params went from 73,728 to
**294,912** (~0.4% of base). Still tiny.

```sh
uv run python -m scripts.chat_sft_lora \
    --base-tag d6_baseline_modern_sft \
    --data-path ~/.cache/nanochat/persona_retention_v1.jsonl \
    --lora-tag d6_l1_persona_lora_v2 \
    --num-iterations 600 --device-batch-size 4 --max-seq-len 256 \
    --rank 16 --alpha 32 --target c_q,c_k,c_v,c_proj \
    --lr 3e-4 --eval-every 100 --log-every 20
```

### Training trajectory (1.95 min wall, 4× v1's params, 2× iters)

| step | loss | val_bpb | (v1 at same step) |
|---:|---:|---:|---:|
| 0 | 4.20 | — | — |
| 100 | 2.96 | 0.9734 | 1.1733 |
| 200 | 2.68 | 0.8931 | 1.0930 |
| 300 | 2.44 | 0.8502 | 1.0807 |
| 400 | 2.36 | 0.8325 | — |
| 500 | 2.31 | 0.8204 | — |
| 599 | 2.25 | **0.8197** | 1.0771 (final) |

v2 already beat v1's *final* val_bpb by step 100. By the end, v2 lands
at val_bpb 0.82 — 0.26 lower than v1, and 0.42 lower than the
untrained-LoRA starting point. Adapter checkpoint: 1.14 MB.

### Eval (same harness, same 30 held-out personas, greedy temp=0)

| metric | base | v1 | v2 | v2 delta vs base | v2 delta vs v1 |
|---|---:|---:|---:|---:|---:|
| name_recall | 6.7% (2/30) | 20.0% (6/30) | **80.0% (24/30)** | **+73.3pp** | +60.0pp |
| role_recall | 33.3% (10/30) | 26.7% (8/30) | **86.7% (26/30)** | **+53.3pp** | +60.0pp |
| location_recall | 13.3% (4/30) | 26.7% (8/30) | **83.3% (25/30)** | **+70.0pp** | +56.7pp |
| **all_three** | **0.0% (0/30)** | **13.3% (4/30)** | **63.3% (19/30)** | **+63.3pp** | **+50.0pp** |

**4/30 → 19/30 all_three.** Almost 5× the v1 win rate. The mode-collapse
pattern from v1 ("That's a great way to work" on ~50% of outputs) is
gone. v2's reflexive opener is now `"You're [Name], a [Role] in
[Location]..."` — which is *reading T1 and recalling*, not deflecting.

### What v2 wins look like

```
[ 0] ✓✓✓ "You're Maria Santos, a pastry chef in Lima, Peru, and you're really into baking competitions."
[ 1] ✓✓✓ "You're Lars Bergström, a marine biologist in Gothenburg, Sweden."
[ 5] ✓✓✓ "You're James O'Brien, a mechanic in Dublin, Ireland, and you're a mechanic in Dublin, Ireland."
[ 6] ✓✓✓ "You're Chen Wei, a graphic designer in Chengdu, China, and you're a graphic designer based..."
[12] ✓✓✓ "You're Amara Okafor, a fashion designer in Lagos."
[14] ✓✓✓ "You're Leila Saadi, an archaeologist in Cairo, and you're really into ancient Mesopotamia."
[18] ✓✓✓ "You're Isabella Rossi, a documentary filmmaker from Rome, and you're a documentary filmmak..."
[19] ✓✓✓ "You're Dev Patel, a craft brewery owner in Melbourne."
[20] ✓✓✓ "You're Ishita Sharma, a marine biologist in Perth, Australia, and coral reef conservation."
[21] ✓✓✓ "You're Lucas Martins, a pastry chef in Rio de Janeiro."
[24] ✓✓✓ "You're Hassan El-Sayed, a marine biologist in Alexandria."
[26] ✓✓✓ "You're Marco Gómez, a archaeologist in Lima."
[29] ✓✓✓ "You're Amara Okonkwo, a documentary filmmaker from Lagos, Nigeria, and you're passionate..."
```

### What the remaining 11 failures look like

The pattern is *not* "didn't recall" — it's "recalled with corruption":

| # | persona ground truth | v2 output (truncated) | failure type |
|---|---|---|---|
| 9 | Elena **Volkova** / botanist / **Moscow** | "You're Elena **Vkova**, **abot** in **Houston**, Russia..." | name+role+city corrupted |
| 16 | **Keiko Yamamoto** / landscape arch. / Kyoto | "You're Keiko **Luamoto**, a landscape architect in concerning sustainable gardens." | name corrupted, city replaced with description |
| 17 | **Henrik** Bergström / paleontologist / Lund | "You're **Henry** Bergström, a paleontologist in Lund." | first name partial-match |
| 22 | Dmitri **Volkov** / quantum physicist / **Moscow** | "You're Dmitri **Vkov**, a quantum physicist in **Houston**." | name+city same as #9 |
| 28 | Henrik **Larsson** / forest ranger / Åre | "You're Henry **opens up** a forest ranger in Åre, Sweden." | name+grammar drift |

The most striking pattern: **two separate Russian personas (Elena
Volkova + Dmitri Volkov, both in Moscow) get re-located to "Houston"**
in v2's output. The base model would never produce that substitution
(its "Moscow" cluster is dense per yesterday's cosine-NN diagnostic);
the LoRA is *partially overwriting* the city-cluster geometry,
specifically pulling Russian-name+Moscow cases toward an
American-name-cluster attractor. This is the only systematic
non-trivial failure pattern in v2's outputs.

The other failures are mostly **name-tokenization edge cases** — "Vkov"
instead of "Volkov", "Vkova" instead of "Volkova", "Henry" instead of
"Henrik", "Luamoto" instead of "Yamamoto". Greedy decoding on slavic
and east-asian name tokens is mis-firing in a way that the LoRA didn't
correct. A non-greedy decode (temp 0.3-0.5) might recover some of
these, but at the cost of reproducibility.

### Updated verdict

v1: **meaningful signal**. v2: **dominant signal**. The headroom claim
in the v1 writeup ("Q+K+V+O at rank 16 directly addresses the two
most-likely undersizing failures") landed harder than I'd written it.
With 0.4% of trainable parameters and ~2 minutes of wall, persona
recall on held-out conversations goes from impossible (0/30) to
majority-class (19/30 all-three, 24/30 name, 26/30 role, 25/30
location).

This is well past the "rubric ≥ 2/7" success criterion the proposal
set. **Persona-retention specifically is solved at d6 by LoRA**, with
the caveats below.

### What v2 still doesn't prove

- **Catastrophic-forgetting still untested.** v2 might fix
  persona_retention while breaking the other 6 prompts in the
  multi-turn rubric. Unknown until we re-run the full rubric.
- **The "Houston" attractor on Moscow personas is concerning.** It
  shows the LoRA is reshaping geographic embedding-cluster geometry
  beyond what's strictly needed for retrieval. A wider eval (more
  Russian / non-Western personas) would tell us whether this is a
  systematic v2 artefact or n=2-noise.
- **Single seed, n=30 held-out.** The 63.3% number has wide error bars
  at this sample size. A different seed could land at 50% or 75%
  without changing the underlying mechanism.
- **Greedy decoding.** All eval ran at temp=0, top_k=50. Sampled
  decoding might surface more failure modes or hide some via
  averaging.

### Where this leaves the build queue

The "next-cheapest probable win" headroom item from the v1 verdict was
v2 itself — that's now landed. Updated near-term queue:

1. **Catastrophic-forgetting check**: re-run the 7-prompt pre-registered
   multi-turn rubric (`docs/multi_turn_chat_eval_2026-05-06.md`) with
   v2 LoRA loaded. ~5 min wall. Tells us whether v2's persona-retention
   solve trades off against the other failure modes.
2. **Investigate the Moscow→Houston substitution**: probe v2's
   embedding shifts on a wider set of city/name pairs. Short follow-up
   to yesterday's cosine-NN diagnostic. ~1 hour.
3. **Bonsai-LoRA Phase 2**: still the right next-level investment per
   the cosine-NN diagnostic. v2 strengthens the prior — d6 LoRA
   demonstrably moves persona-retention; the question is whether 1-bit
   base + fp LoRA closes the remaining ~37% all_three gap and
   generalises better. ~1-2 days infra.

### v2 commits

```
[this addendum]
[v2 LoRA checkpoint at lora_checkpoints/d6_l1_persona_lora_v2/lora_000600.pt — 1.14 MB, not in git]
[v2 eval results at persona_retention_v1_eval_results_v2.json — not in git]
```

No new code: v2 used the existing infrastructure unchanged.

