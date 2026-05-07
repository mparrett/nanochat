# Cosine-NN diagnostic on d6_stage2_pretrain_s1_sft — 2026-05-06

**Status**: side-question follow-up to
`docs/cosine_nn_diagnostic_2026-05-06.md`. Probe re-run on the Stage 2
SFT checkpoint (Hope/NL learned-gate memory at L3) with the same
17-token probe set. **The dominant-axis verdict is unchanged**:
Stage 2's recurrent memory does *not* materially shift the FP /
binary distribution that the diagnostic measures. Two qualitative
deltas worth flagging.

## Side question (from yesterday's HANDOFF)

> Re-run cosine-NN on `d6_stage2_pretrain_s1_sft`'s embeddings to test
> whether internal recurrent memory shifts the axis distribution
> (e.g., does Stage 2 specifically attenuate right-context drift?).
> ~1 minute marginal compute; intrinsically interesting whether or
> not Stage 2 ever shows multi-turn improvement.

## Method

Same `dev/cosine_nn_probe.py`, parameterised today with `--model-tag`.
Identical 17-token probe set, identical top-10 cosine-NN reporting.
Stage 2 checkpoint loaded as `chatsft/d6_stage2_pretrain_s1_sft` (step
375). Embedding matrix shape `(32768, 384)` matches baseline. ~30 s
on M2 mps. Single run; embedding matrix is deterministic.

```
uv run python -m dev.cosine_nn_probe --model-tag d6_stage2_pretrain_s1_sft
```

Raw run log: `/tmp/cosine_nn_stage2.log` (committed inline below).

## Per-token deltas vs baseline

Baseline classifications repeated in italics for comparison; bold flags
real qualitative differences.

| token | baseline NN top-3 | stage 2 NN top-3 | delta |
|---|---|---|---|
| `'Alex'` | _Michael, James, David_ | Michael, James, Alex (self-leading-space) | sharper, same axis |
| `'engineer'` | _scientist, homeowner, actor_ | photographer, researcher, scientist | **purer occupations** (no homeowner/actor noise) |
| `'software'` | _equipment, hardware, clothing_ | technology, machinery, hardware | sharper categorical mass-nouns |
| `'Java'` | _JavaScript, Python, Utah_ | JavaScript, Windows, PHP | **less overloaded** — places/brands gone, mostly tech |
| `'J'` | _" J", j, -J_ | -J, " J", j | identical pattern (left-context typographic) |
| `'Sydney'` | _China, Georgia, London_ | Pakistan, Victoria, Phoenix | same axis (places) |
| `'Australia'` | _Lanka, NASA, Tesla_ | India, frica, Europe | **sharper** — countries cluster, less brand/person noise |
| `'5'` | _7, 3, 6_ | 3, 7, 6 | identical (digits cluster, very strong) |
| `'apples'` | _oranges, names, shoes_ | beans, cookies, bananas | sharper (foods, no "names" intrusion) |
| `'left'` | _right, first, start_ | right, fruit, cules | similar (mostly antonym + noise) |
| `'the'` | _" the", " The", "The"_ | " the", "The", " The" | identical (left-context typographic) |
| `'store'` | _enclosure, ulate, clinic_ | store(self), purchase, company | slightly cleaner; mostly right-context |
| `'too'` | _" too", oooo, amboo_ | " too", "Too", isa | same pattern (left-context phonetic) |
| `'Portland'` | _Austin, Melbourne, Seattle_ | Tokyo, Austin, Ontario | sharper city cluster |
| **`<\|python_start\|>`** | _'Ma', ',[', ' YOU'_ (sims 0.27-0.37) | 、, ($, learn (sims 0.24-0.29) | **bracket cluster scrambled** |
| `<\|output_start\|>` | _' Ex', ' Mem', ' Aut'_ | hysical, identifying, delectable | both underspecified, different fragments |
| `<\|assistant_start\|>` | _'+\n', '---\n\n', '–\n'_ | '**\n\n', ".'\n", '!!\n' | same pattern (newline delimiters) |

## Aggregate axis distribution

| axis | baseline | stage 2 | delta |
|---|---:|---:|---|
| right-context drift (FP-flavored) | 11 / 17 (~65 %) | 10 / 17 (~59 %) | within 17-token sample noise |
| left-context collision (binary-flavored) | 3 / 17 (~18 %) | 3 / 17 (~18 %) | identical (J, the, too) |
| mixed / underspecified | 3 / 17 (~18 %) | 4 / 17 (~24 %) | `<\|python_start\|>` moved here |

Re-classifications:

- **`<|python_start|>` baseline → right-context (clean bracket
  cluster) → stage 2 → mixed/underspecified.** Strongest
  single qualitative delta. Stage 2's NNs include some bracket-y tokens
  (`(`$`, `[`, `$\n`, `-\n`) but mixed with semantic words
  (`learn`, `grandparents`, `politicians`, `couples`, `delectable` is
  in the `<|output_start|>` row — yes, both special tokens look like
  the embedding hasn't found a clean cluster). Cosine similarities also
  drop ~10 % (top NN 0.37 → 0.29). The GSM8K bracket-template trigger
  geometry is partially decoupled.

The 11→10 change in right-context count is `<|python_start|>` moving
to mixed; everything that was right-context in baseline is still
right-context in Stage 2.

## Two qualitative observations beyond the axis count

### 1. Stage 2 sharpens several right-context clusters

`engineer`, `Australia`, `Portland`, `apples`, `software`, `Java` all
have *cleaner* categorical neighbours under Stage 2:

- `engineer`: drops `homeowner`, `actor`; gains `photographer`, `researcher`, `biologist`, `designer`. Pure occupations.
- `Australia`: drops `NASA`, `Tesla`, `microprocessor`, `Roosevelt`, `CDC`; gains `India`, `Europe`, `Italy`, `Philippines`, `Arabia`, `America`, `China`. Pure countries.
- `Portland`: drops `Ghana`, `Rome`, `ville`; gains `Tokyo`, `Ontario`, `Quebec`, `Louisiana`, `Kansas`. Pure cities.
- `Java`: drops places (`Utah`, `Austin`) and brands (`BMW`, `Tesla`); gains `Windows`, `PHP`, `pandas`, `ChatGPT`, `Math`, `Geometry`. Mostly tech with some residual ambiguity.

This is consistent with the L3 memory module having a *narrowing*
effect on category-specific embedding neighbourhoods. The dominant-
axis diagnosis is unchanged, but the specific clusters are tighter.
The mechanism for "Sydney attractor in capital-cities state" is
*identical* under Stage 2 (places still cluster densely with Sydney);
the cluster composition is just slightly different.

### 2. Stage 2 specifically scrambles `<|python_start|>`'s template association

This is the most surprising single result and the most worth flagging.

Baseline `<|python_start|>` was embedded as *"the thing that comes
right before bracket-prefixed Python syntax"* — a clean, FP-classic
right-context cluster. The math-mode reflex on number-shaped input
went through this proximity directly.

Stage 2's `<|python_start|>` cluster is much noisier and weaker:

```
'、'(0.29), ' ($'(0.28), 'learn'(0.28), ' grandparents'(0.24),
'"-'(0.24), '['(0.24), ' -\n'(0.24), ' politicians'(0.24),
'$\n'(0.24), ' couples'(0.24)
```

Bracket / paren / dollar tokens are still represented (`($`, `[`,
`-\n`, `$\n`, `"-`) but they share the cluster with random semantic
words. Top similarity drops 0.37 → 0.29. By the same diagnostic, this
template-trigger embedding has been *partially decoupled* by Stage 2's
recurrent memory training.

`<|output_start|>` was already underspecified in baseline; Stage 2
moves it to a different underspecified cluster but the diagnosis is
the same.

`<|assistant_start|>` is *unchanged* (still cleanly right-context as a
"newline delimiter" — the cluster composition is even nearly identical).

So the math-mode reflex's geometry is the *one* mechanism Stage 2
appears to disturb in this probe. This wasn't on either of our prior
prediction lists for what Stage 2 would do. **Whether it actually
manifests as fewer math-mode triggers in chat outputs is a separate
question, not answered by this diagnostic.** It's a hypothesis, not a
finding.

## Verdict on the side question

**Does Stage 2's recurrent memory shift the FP / binary axis
distribution?** No, not materially. ~65 % → ~59 % right-context is
within 17-token sample noise; the left-context fraction is unchanged
(same three tokens: J, the, too); the mixed fraction grew by one
specifically because `<|python_start|>` got scrambled.

**Implication for Bonsai-LoRA priority**: unchanged. Codebook
coarsening is still the indicated fix for the dominant axis. The
Stage 2 architecture doesn't address the FP-drift mechanism — it
sharpens some clusters and scrambles one template trigger. The
dominant-axis diagnosis would arrive at the same fix direction for
either model.

**Implication for capacity-bound left-context**: also unchanged. The
same three tokens (J, the, too) collide identically. Adding capacity
(M4 / d8+) is still the indicated fix for this axis; Stage 2's L3
memory is not capacity in the relevant sense.

## What would change the verdict

This probe is 17 tokens, judgment-call classification, single seed.
The aggregate count (11 vs 10 right-context) is sample-noise-bounded.
A larger stratified probe (200+ tokens by frequency / POS / role)
might either (a) confirm the small attenuation is real and Stage 2 is
nudging the distribution in a useful direction, or (b) confirm it's
sample noise. Neither outcome would change the *dominant-axis*
verdict; both would tighten the *magnitude* claim.

The `<|python_start|>` scrambling, on the other hand, is a strong
single-token signal. If we wanted to follow it up: probe a wider set
of GSM8K-template tokens (`<|output_start|>`, digit-prefix bytes,
bracket variants, structured-task delimiters) under both models, and
report whether Stage 2 systematically breaks template-trigger geometry
or only this one token. ~5 minutes more compute. Optional.

## Limits

- Same 17-token probe size as baseline; sample-noise floor on aggregate counts.
- Same eyeball-classification methodology; reasonable people might re-classify a couple of mixed tokens differently.
- Embedding-matrix probe only — doesn't measure how the L3 recurrent memory module itself reshapes hidden states at runtime; only the input/output token embedding geometry post-SFT.
- One Stage 2 checkpoint (the original `s1` schedule). Hasn't been re-run on `d6_stage2_pretrain_s2_sft` (s2 schedule), if that checkpoint is still on disk.

## Artefacts

- Probe code: `dev/cosine_nn_probe.py` (now accepts `--model-tag`).
- Comparison reference: `docs/cosine_nn_diagnostic_2026-05-06.md`.
- Source framework: trx4mr Phase 5 retro
  (`~/projects-new/trx4mr/experiments/blabberverse-phase5-impl-and-takeaways.html`,
  §2.4 "Probe geometry, then debug with a two-axis framework").
- Reproducibility: ~30 s on M2 mps. Single run; embedding matrix is deterministic.

## Next session — fold-into queue

- This result should be referenced in the LoRA proposal (`docs/lora_proposal_2026-05-06.md`, L1-d6 vs L1-bonsai A/B framing) as additional support for *not* expecting Stage 2 to be the dominant-axis fix on its own.
- The `<|python_start|>` scrambling deserves a one-line addition to the FD proposal (`docs/feedback_descent_proposal_2026-05-06.md` Direction B): Stage 2 may attenuate template triggers in a way orthogonal to its multi-turn-rubric performance; worth a math-mode-reflex eval on Stage 2 outputs specifically (count `<|python_start|>` emission rate on number-shaped chat prompts under both models — easy comparison).

## Raw probe output for archival

```
Embedding matrix: torch.Size([32768, 384])
Device: mps:0, dtype: torch.float32
Model tag: d6_stage2_pretrain_s1_sft

token                         tid  kind            top-10 cosine NNs
'Alex'                      24372  special         'Michael'(0.48), 'James'(0.46), ' Alex'(0.43), 'Mike'(0.41), 'aniel'(0.40), 'David'(0.39), 'John'(0.39), 'Paul'(0.38), ' Chris'(0.38), 'ily'(0.38)
'engineer'                  10131  leading-space   ' photographer'(0.51), ' researcher'(0.47), ' scientist'(0.44), ' biologist'(0.44), ' designer'(0.43), ' gardener'(0.42), ' trainer'(0.42), ' lecturer'(0.41), ' chemist'(0.41), ' surgeon'(0.41)
'software'                   2775  leading-space   ' technology'(0.43), ' machinery'(0.42), ' hardware'(0.41), ' equipment'(0.41), ' curriculum'(0.40), ' infrastructure'(0.39), ' instrumentation'(0.39), 'Software'(0.39), 'ware'(0.38), ' jewellery'(0.36)
'Java'                       8959  leading-space   ' JavaScript'(0.47), ' Windows'(0.39), ' PHP'(0.38), ' pandas'(0.37), ' ChatGPT'(0.35), ' nylon'(0.34), ' Math'(0.33), ' Pinterest'(0.33), ' Vienna'(0.33), ' Geometry'(0.33)
'J'                            74  special         '-J'(0.52), ' J'(0.52), 'j'(0.47), '.J'(0.38), ' j'(0.35), 'ELL'(0.34), ' BL'(0.33), ' Jac'(0.32), ' Shell'(0.32), ' Jud'(0.32)
'Sydney'                    15293  leading-space   ' Pakistan'(0.44), ' Victoria'(0.43), ' Phoenix'(0.43), ' Indonesia'(0.41), ' Utah'(0.40), ' Atlanta'(0.40), ' Vancouver'(0.40), ' Bristol'(0.39), 'Rose'(0.39), ' Mercury'(0.39)
'Canberra'                      -  multi-token (3 tokens)  [skipped]
'Australia'                 30246  special         ' India'(0.44), 'frica'(0.42), ' Europe'(0.40), ' humanity'(0.39), 'ennessee'(0.39), ' Italy'(0.39), ' Philippines'(0.38), ' Arabia'(0.37), ' America'(0.37), 'China'(0.37)
'5'                            53  special         '3'(0.61), '7'(0.61), '6'(0.61), '4'(0.59), '9'(0.58), '2'(0.55), '0'(0.52), '8'(0.52), '50'(0.48), '10'(0.44)
'apples'                    10642  leading-space   ' beans'(0.39), ' cookies'(0.39), ' bananas'(0.38), ' peas'(0.38), ' oranges'(0.38), ' lipids'(0.36), ' sandwiches'(0.35), ' reagents'(0.35), ' ounces'(0.35), ' eggs'(0.34)
'left'                      15430  special         'right'(0.40), 'fruit'(0.32), 'cules'(0.31), 'mate'(0.31), ' α'(0.29), 'first'(0.29), ' anode'(0.29), 'bite'(0.29), 'four'(0.29), 'ipse'(0.29)
'the'                        1235  special         ' the'(0.66), 'The'(0.62), ' The'(0.59), '-The'(0.56), ':The'(0.53), ',the'(0.46), '—the'(0.45), '.The'(0.44), '"The'(0.42), '“The'(0.39)
'store'                     25924  special         ' store'(0.41), ' purchase'(0.37), ' company'(0.35), ' sequence'(0.32), 'issue'(0.32), ' yields'(0.31), ' regions'(0.31), 'care'(0.30), 'block'(0.30), ' harbor'(0.30)
'too'                       27230  special         ' too'(0.49), 'Too'(0.38), 'isa'(0.35), 'no'(0.34), 'o'(0.33), 'ello'(0.33), 'uity'(0.32), ' Too'(0.32), 'via'(0.31), 'iga'(0.31)
'Portland'                  18231  leading-space   ' Tokyo'(0.43), ' Austin'(0.40), ' Ontario'(0.40), 'adelphia'(0.39), ' Quebec'(0.39), ' Glasgow'(0.38), ' Atlanta'(0.37), ' Louisiana'(0.37), ' Melbourne'(0.37), ' Kansas'(0.37)
'<|python_start|>'          32764  special         '、'(0.29), ' ($'(0.28), 'learn'(0.28), ' grandparents'(0.24), '"-'(0.24), '['(0.24), ' -\n'(0.24), ' politicians'(0.24), '$\n'(0.24), ' couples'(0.24)
'<|output_start|>'          32766  special         'hysical'(0.29), ' identifying'(0.29), ' delectable'(0.27), ' obvious'(0.26), 'aneous'(0.26), ' recognised'(0.26), ' gross'(0.25), ' Secondary'(0.25), ' thermodynamic'(0.25), 'Background'(0.24)
'<|assistant_start|>'       32762  special         '**\n\n'(0.39), ".'\n"(0.38), '!!\n'(0.38), '?\n'(0.37), '–\n'(0.37), ':\n'(0.37), '!"\n\n'(0.37), '.”\n'(0.35), ' –\n'(0.35), '!\n'(0.35)
```
