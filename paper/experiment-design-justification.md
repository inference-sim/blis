# Why These ~50 Experiments?

BLIS predicts LLM inference latency without GPUs. These experiments validate that prediction across the configurations production operators actually deploy.

---

## The Question We're Answering

**Can BLIS replace GPU benchmarking for capacity planning?** To answer yes, we need ground-truth measurements spanning diverse models, hardware, workloads, and serving configurations. The matrix is designed to be *minimal* (every experiment maps to a figure) and *sufficient* (covers the axes that matter).

---

## What We Vary (4 Axes)

### Models (8) — architecture diversity, not model zoo padding

We pick along two axes that change the simulator's code paths: **dense vs. MoE** and **parameter scale**.

| Category | Models | What it tests |
|---|---|---|
| Small dense (TP=1) | Llama-2-7B, Llama-3.1-8B, Qwen3-14B | Roofline FLOPs/bandwidth estimation — no parallelism confounds |
| Medium dense (TP=2) | CodeLlama-34B | Tensor-parallel communication overhead |
| Large dense (TP=4) | Llama-2-70B | TP scaling limits (GQA with 8 KV heads) |
| MoE (sparse activation) | Mixtral-8x7B, Mixtral-8x22B | Sparse compute vs. full weight memory footprint |
| MoE + FP8 | Llama-4-Scout-17B-16E | Quantized weights + 16-expert topology |

**Vidur overlap (deliberate).** Three of our models — Llama-2-7B, Llama-2-70B, CodeLlama-34B — are shared with [Vidur](https://github.com/microsoft/vidur) (Microsoft, NSDI '25), which supports {Llama-2-7B, Llama-2-70B, CodeLlama-34B, Llama-3-8B, Llama-3-70B, InternLM-20B, Qwen-72B} on {A40, A100, H100}. This overlap enables direct apples-to-apples accuracy comparison on the same model×hardware pairs. Our matrix then extends beyond Vidur's coverage with MoE models (Mixtral-8x7B, Mixtral-8x22B, Llama-4-Scout), FP8 quantization, expert parallelism, L40S hardware, and config knob sweeps — none of which Vidur evaluates.

**Excluded:** Vision-language models (different compute graph), 405B+ (same code path as 70B, 4× the GPU cost).

### Hardware (3) — bandwidth tiers that exist in datacenters

| GPU | Bandwidth | Role |
|---|---|---|
| H100-80GB | 3.35 TB/s, NVLink | Primary — most experiments here |
| A100-80GB | 2.0 TB/s, NVLink | Cross-generation validation (measured 1.6-1.7× slower than H100, matching the bandwidth ratio) |
| L40S-48GB | 864 GB/s, PCIe | Budget tier — tests viability limits |

Three GPUs = three distinct (bandwidth × interconnect) classes. Adding H200 or A10G covers no new class.

### Workloads (4) — from ServeGen traffic profiles

These four workloads are drawn from [ServeGen](https://arxiv.org/abs/2401.09056) traffic profiles, each representing a structurally distinct serving regime:

| Workload | Input | Output | Traffic pattern | What it stresses |
|---|---|---|---|---|
| General | 200-800 | ~250 | Highly variable burstiness, temporal shifts | Auto-scaling, dynamic resource allocation — tests the simulator under non-stationary arrival rates |
| Codegen | 500-1500 | ~250 | Development-cycle patterns, template-based outputs | Structured generation with long prefills — stresses chunked-prefill batch formation |
| Roleplay | 100-400 | ~250 | Smoother, human-paced interaction | Conversational serving with steady load — baseline for prefill/decode balance |
| Reasoning | 934 | ~1448 | Bimodal output (reasoning + answer tokens) | Long-output decode with 6× longer batch occupancy — stresses KV cache capacity and preemption |

Together they cover the (input_length, output_length, burstiness) space: short-input steady (Roleplay), long-input bursty (General, Codegen), and long-output bimodal (Reasoning). This ensures the simulator is validated across the traffic shapes that actually differ in how they load prefill vs. decode and how they pressure KV cache memory.

### Config Knobs (5) — what operators actually tune

`max_num_batched_tokens` (1K/2K/8K), `cpu_offloading` (on/off), `gpu_memory_utilization` (0.9/0.95), tensor parallelism (default/2×), expert parallelism (MoE only: 1/2/4).

---

## Why ~50, Not Thousands?

Full factorial = 8 × 3 × 4 × many knob combos = thousands. Three pruning rules get us to ~50:

1. **Viability pruning.** Large models don't fit on small GPUs. Validated with capacity formulas → 18 viable model×GPU combos (down from 24).

2. **Config sweeps on 2 models only.** Knob sensitivity depends on architecture class (dense vs. MoE), not model identity. We sweep on the cheapest model per class: Llama-3.1-8B (1 GPU) and Mixtral-8x7B (2 GPUs). Sweeping Llama-2-70B instead would cost 4× the GPU-hours for the same information. Saves ~70 experiments.

3. **Cross-hardware = General workload only.** Hardware scaling is workload-independent (confirmed: A100/H100 ratio is 1.6-1.7× regardless of workload). Running all 4 workloads on A100 and L40S would be redundant. Saves ~21 experiments.

---

## Every Experiment Maps to a Figure

| Figure | Claim | Experiments | Key comparison |
|---|---|---|---|
| **Fig 1: Model comparison** | Accurate across 7B-141B, dense + MoE | 7 | Architecture is sole variable |
| **Fig 2: Hardware comparison** | Bandwidth-based roofline generalizes across GPUs | ~16 | Same model, different GPU |
| **Fig 3: Workload comparison** | Predictions hold across token distributions | ~28 | Same model+GPU, different workload |
| **Fig 4: Config sensitivity** | Captures impact of operator-tunable knobs | 13 | Same model, one knob varied |

Zero experiments exist outside these four figures.

---

## Phase Ordering: Publishable at Any Stopping Point

GPU time is expensive. The matrix is ordered so you can stop early and still have results.

| Stop after | Experiments | What you have |
|---|---|---|
| Phase 0-3 | 21 | 5 models × 3 workloads on H100 (Figures 1, 3) |
| Phase 0-5 | 32 | + config sweeps (Figure 4 complete) |
| Phase 0-8 | 44 | + A100/L40S cross-hardware (Figure 2 complete) |
| Phase 0-9 | ~50 | + Reasoning workload (full matrix) |

---

## Measurement Rigor

- **Sub-saturation by design.** Every experiment runs below 50% of calibrated throughput limit. At saturation, latency is dominated by queueing (tests the scheduler, not the latency model). Sub-saturation isolates what we're actually validating.
- **Single-variable comparisons.** Each experiment varies one dimension from a per-model baseline (default TP, mbt=2048, no offload, gpu_mem=0.9).
- **Stock vLLM v0.15.1.** No custom fork. Reproducible by anyone.
- **Pre-calibration.** Every model×GPU×TP combo has a measured per-token decode latency. Experiments that exceed the safe rate are flagged and rate-capped.

---

## vs. Prior Work

| System | Models | GPUs | Workloads | Config sweeps | MoE | FP8 |
|---|---|---|---|---|---|---|
| Vidur (NSDI '25) | 3 dense | 1 (A100) | 2 | No | No | No |
| Splitwise (ISCA '24) | 2 dense | 1 (A100) | 1 | No | No | No |
| DistServe (OSDI '24) | 3 dense | 2 (A10G, A100) | 2 | Partial | No | No |
| **This work** | **5 dense + 3 MoE** | **3 (H100, A100, L40S)** | **4** | **5 knobs, 11 sweeps** | **Yes** | **Yes** |

Three of our dense models (Llama-2-7B, Llama-2-70B, CodeLlama-34B) and two GPUs (A100, H100) overlap with Vidur, enabling head-to-head accuracy comparison on shared configurations. Our matrix then extends into territory no published simulator evaluation covers: MoE architectures, FP8 quantization, expert parallelism, and systematic config knob sensitivity.

---

## Known Limitations

- **vLLM only** — doesn't validate against TensorRT-LLM or SGLang (different schedulers, future work)
- **Single-node** — max 8 GPUs, no pipeline parallelism across nodes
- **Synthetic workloads** — controlled token distributions, not production traces (by design: isolates latency model accuracy from arrival process noise)
- **FP8 only for MoE** — reflects current practice where FP8 is used to fit models that wouldn't otherwise be viable

---

## Bottom Line

~50 experiments. 8 architectures (7B-141B, dense + MoE + FP8). 3 GPU tiers. 4 workloads. 5 serving knobs. Every experiment maps to a figure. Publishable results at any early-stopping phase. Broader than any published simulator evaluation we're aware of.
