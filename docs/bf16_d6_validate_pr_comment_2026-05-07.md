Ran a horizon-scale A/B to back this up. Same d6 recipe (~74M params, `num_iterations=5000`), single delta `NANOCHAT_DTYPE=bfloat16`, fp32 baseline as reference. Both on an M2 via MPS.

bf16 lands at **val_bpb 1.169138** vs the fp32 baseline's **1.168621** at step 5000 — a gap of **+0.0005 (0.04%)**, well under the per-eval sampling noise on a 524K-token val budget. The descent through warmdown was smooth, no instability spikes, no NaN. At this scale bf16 costs nothing in quality.

The wins on the side:
- checkpoints 34% smaller (185 MB vs 281 MB)
- unblocks `device_batch_size=48` on M2, which OOMs in fp32

Validating end-to-end surfaced a second mixed-dtype boundary on MPS, the same shape as the one this PR fixes. SDPA in `flash_attention.py` rejects bf16 `q` against the fp32 KV cache — CUDA implicitly promotes, MPS hard-rejects:

```
RuntimeError: Expected query, key, and value to have the same dtype,
but got query.dtype: c10::BFloat16 key.dtype: float and value.dtype: float
```

Fixed by casting k/v to `q`'s dtype at the SDPA boundary — dtype-neutral, leaves the CUDA FA3 path untouched. Happy to send that as a follow-up PR if it's useful.
