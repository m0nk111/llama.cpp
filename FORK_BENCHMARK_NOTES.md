# Fork benchmark notes — cuda128-laguna-tq-full

Hardware: RTX 3060 12 GB (sm_86, GPU0) + RTX 5060 Ti 16 GB (sm_120, GPU1),
PCIe, DDR4 host RAM. Build: `build-cuda128-full` (CUDA 12.8, GGML_CUDA_FA=ON,
arch 86;120), llama-server b10553 (25af3df71). Measured 2026-08-10 via
llama.cpp `llama_cpp_guardian/scripts/bench_qwen36_variants.py`; raw results in
`llama_cpp_guardian/data/bench-qwen36/`.

## Qwen3.6-35B-A3B (MoE, ~40 layers), i1-Q4_K_M, ctx 32768, greedy, median of 3

| Variant | Prompt t/s | Gen t/s | vs baseline |
|---|---:|---:|---:|
| q4_0 KV, no draft (baseline) | 2261 | 80.3 | 1.00x |
| **turbo4 KV, no draft** | 2239 | **89.1** | **1.11x** |
| q4_0 KV + DFlash draft (BF16) | 1614 | 54.6 | 0.68x |
| turbo4 KV + DFlash draft | 1608 | 58.3 | 0.73x |

### Finding 1: DFlash only pays off when decode is bandwidth-starved

- On **Laguna-S-2.1** (118B MoE, UD-IQ4_XS, `ngl 16` -> most layers in host
  RAM, decode ~0.12 tok/s, severely DDR4-bound): DFlash **doubled** gen speed
  (0.12 -> 0.24 tok/s) even at acceptance 0.167 (6/36).
- On **Qwen3.6-35B** (fully GPU-resident, 80 tok/s decode): the same DFlash
  approach **costs 32%** (80.3 -> 54.6 tok/s) at acceptance ~0.21 (53/248,
  mean accepted 2.7 of n_max 8). Prompt processing also drops ~29%.

Conclusion: with a GPU-resident target, the per-step draft+verify overhead
exceeds the savings unless draft acceptance is much higher than ~0.2. DFlash
(and speculative decoding in general on this stack) earns its keep only when
the target decode is slow enough (host-RAM/offload-bound) that drafting
amortizes. Track draft acceptance in server logs (`draft acceptance = ...`)
before enabling it per model.

### Finding 2: turbo4 KV is a free ~11% decode win here

turbo4 KV (with fork auto-asymmetric K->q8_0 upgrade at GQA 8:1) beats q4_0 KV
by ~11% on decode at equal prompt speed. No quality anomalies observed in
benchmark outputs. This is now the production KV type for this model.

## Artifacts

The mmproj vision tower offloads independently (`mmproj_use_gpu`, on by
default) and does not consume the text model's `-ngl` budget. The old habit of
running vision at `ngl 40` (one layer + embeddings on CPU) cost ~10x decode
speed (9.5 tok/s) for no VRAM reason once the KV cache is turbo4.

For Qwen3.6-35B + mmproj at full 262144 ctx, all layers fit on GPU at
`ngl 99` with `-ts 0.34,0.66` -> **99.7 tok/s vision gen** (was 9.5).

Caveat: the equal split `0.38,0.62` loads fine but **segfaults on the first
vision request** (GPU0 compute-buffer OOM, 248 MiB short). Leave >= 1 GB
headroom on the smaller GPU for the vision compute buffer; the mmproj warmup
at load time does not exercise this, only a real image request does.

### Finding 4: ctx beyond `n_ctx_train` needs a server patch + smaller batch

Upstream llama-server hard-caps per-slot context at the model's training
context (`server-context.cpp`: "the slot context exceeds the training context
of the model - capping"), which silently nullifies `-c` above `n_ctx_train`
even when rope scaling is configured. This fork now skips that cap when
`rope_scaling_type` is set (yarn-ctx graft).

Qwen3.6-35B at `-c 393216` (1.5x native) + YaRN 1.5x on this host:

- 292,959-token needle-in-haystack: PASS (811 tok/s prefill, 17.2 tok/s decode
  at ~293k depth, needle found verbatim).
- Requires `--batch-size 1024 --ubatch-size 256`: at default 2048/512 the
  1.6 GB compute buffer fails to allocate on GPU1 and the server falls back
  to non-pipelined execution. Note the failure is recoverable-from — the
  server still reports healthy — so check logs for "compute buffer allocation
  failed" after any ctx bump.
- `rope.freq_base = 10^7` on this model family makes YaRN quality loss at
  1.5x negligible for retrieval tasks; generative quality beyond 262144 is
  not yet characterized.

## Artifacts

- Benchmark script: `llama_cpp_guardian/scripts/bench_qwen36_variants.py`
- Raw results: `llama_cpp_guardian/data/bench-qwen36/results.json` +
  per-variant server logs
- Guardian config using these findings:
  `llama_cpp_guardian/config/models.yaml` (`Qwen3.6-35B-A3B-HauhauCS-Aggressive-Turbo4`)
