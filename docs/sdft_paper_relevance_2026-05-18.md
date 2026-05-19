# SDFT (Self-Distillation Fine-Tuning) — Relevance to nanochat / Hope-NL

**Paper:** Shenfeld, Damani, Hübotter, Agrawal. *Self-Distillation Enables Continual Learning.* MIT / Improbable AI / ETH Zurich. arXiv:2601.19897, Jan 2026.
**Code/data:** http://idanshenfeld.com/SDFT
**Source:** Bear note "Web Clip - SelfDistillation Enables Continual Learning pdf" (2026-05-16), local PDF at `~/Library/Mobile Documents/com~apple~CloudDocs/Downloads/self-2601.19897.pdf`.

## What SDFT is

Drop-in replacement for SFT that mitigates catastrophic forgetting without needing a reward function. The model is used in two roles simultaneously:

- **Teacher:** the same model conditioned on a demonstration `c` and the prompt `x`, producing `Q = π(·|x, c)`.
- **Student:** the model conditioned only on `x`, producing `P = π_θ(·|x)`.

Training samples rollouts from the **student**, then minimizes reverse-KL `D_KL(π_θ(·|x) || π(·|x, c))` token-wise on those rollouts (EMA of the student parameters for the teacher). On-policy, no reward function, no separate teacher model.

Theoretical framing (§3.1): the objective is equivalent to maximizing an implicit reward `r(y, x, c) = log π(y|x,c) − log π(y|x)` derived from the model's own ICL behavior. Rests on the **In-Context Assumption** (Eq. 4): `π*_{k+1}(y|x) ≈ π(y|x, c)` — i.e., conditioning on a single demonstration produces a near-optimal policy.

## Headline results (Qwen2.5-7B-Instruct)

- **Skill learning** (Science Q&A, ToolAlpaca, Medical): Pareto-better than SFT on the new-task-accuracy ↔ prior-tasks-retention plane (Fig 4). SFT drops prior-task performance ~10 pp; SDFT loses ~0–2 pp.
- **Knowledge acquisition** (Wikipedia disasters, 200K tokens): on injected factual knowledge, SDFT hits **89 strict / 100 lenient / 98 OOD** vs SFT's **80 / 95 / 80**, nearly matching oracle-RAG on strict accuracy (Table 1).
- **Sequential continual learning** (Tool → Science → Medical, Fig 3): SDFT retains all three; SFT catastrophically forgets the earlier two.
- **Teacher-base divergence:** SFT model drifts 1.26 nats from base policy; SDFT teacher only 0.68 nats. The "wiser" teacher stays close to the base, which is precisely the trust-region condition.

## Does it apply to us?

Three threads.

### 1. Scale is the hard blocker for porting

The entire mechanism depends on Eq. 4 — that prepending a demonstration `c` actually shifts `π(·|x, c)` toward an optimal policy. At Qwen-7B scale, ICL is strong enough that this works. At **d6 (~tens of M params)** or even **d20**, few-shot ICL is weak-to-nonexistent: the teacher won't be wiser than the student, the implicit reward signal collapses, and the method degrades to noise (or worse, drives the student toward a noisy conditional). Before any port attempt, the prerequisite is an empirical ICL check on the base model: does conditioning on a demonstration measurably improve task accuracy? If no, SDFT cannot work at this scale.

ICL emerges somewhere in the 1B–7B range depending on data and architecture. nanochat's depth ladder is well below this threshold.

### 2. Direct overlap with the δ-mem coding result

The paper's tool-use experiment (ToolAlpaca) and its skill-learning Pareto plot are structurally identical to the δ-mem MBPP setup: a new skill is added, prior capabilities are tracked, and the win is reported as "more new-skill accuracy at less prior-capability cost." The δ-mem field result (`delta_mem_field_result_2026-05-18.html`) showed +8 pp on MBPP — SDFT reports similar OOD gains on tool-use.

This is a **named alternative explanation** for the δ-mem coding generalization story: maybe a better training recipe (SDFT, not architectural memory) achieves the same Pareto improvement. Doesn't invalidate δ-mem — different mechanism, different axis — but a reviewer would reasonably ask "what does plain SFT-with-SDFT look like as a baseline?". Worth addressing in any writeup that emphasizes "δ-mem reduces catastrophic forgetting on coding."

Practical implication: if we ever run on Qwen-7B-class models (which we already do for the δ-mem MVE / quantization work), SDFT is the natural recipe-side baseline to compare δ-mem against.

### 3. Continual-learning framing isn't our current problem

SDFT's headline win (Fig 3) is sequential skill acquisition without regression. Our pipeline is single-shot: pretrain → SFT → eval. Catastrophic forgetting only becomes a primary concern if we start stacking training stages (e.g., SFT → domain SFT → RL → more SFT, or repeated fine-tunes on the same checkpoint). At that point, SDFT is the obvious recipe to reach for. Until then, the continual-learning angle is academic.

## Decision

**Do not port to nanochat.** ICL-at-small-scale assumption is fatal. Keep it on the radar as:

1. The standard reference for "SFT replacement that preserves prior capabilities at scale."
2. The named recipe-side alternative explanation for the δ-mem MBPP gain. Address in the δ-mem writeup before any external review, even if only as a one-paragraph "we don't run SDFT as a baseline because [scale / scope]" disclaimer.
3. A future baseline if/when we benchmark δ-mem on Qwen-7B or other base models with usable ICL.

## Method sketch (for future reference)

```
For each training step:
  Sample (x, c) where c is a paired demonstration for prompt x.
  Teacher distribution Q = π_θ_EMA(·|x, c)   # same model, EMA params, conditioned on c
  Student rollout y ~ π_θ(·|x)               # student-generated, on-policy
  Loss = Σ_t log[π_θ(y_t | y_<t, x) / π_θ_EMA(y_t | y_<t, x, c)]   # reverse-KL on rollout tokens
  Update θ; update EMA.
```

Teacher prompt template (verbatim from §3):

```
<Question>
This is an example for a response to the question:
<Demonstration>
Now answer with a response of your own, including the thinking process:
```

Data requirement that differs from standard SFT: need **paired (prompt, demonstration)** examples where the demonstration is a *different* example of the same skill, not the gold response for the prompt itself. Standard SFT corpora (SmolTalk, etc.) provide `(prompt, response)`; SDFT needs the response of some *other* in-skill example as `c`. Either retrieve at training time or precompute pairings.
