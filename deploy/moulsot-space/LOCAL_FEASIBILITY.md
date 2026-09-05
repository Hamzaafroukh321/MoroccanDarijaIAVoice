# Local MoulSot feasibility — September 5, 2026

**Later update:** the BF16-only assessment below has been superseded by a
successful isolated quantized experiment. MoulSot Q4_K_M + Q8 audio projector
loaded on this laptop without changing Python pins. One model-resident GPU
request took 1.586 s for a 7.207 s clip; this is a single diagnostic, not native
accuracy or an average latency. See
[measured local results](../../bench/results/local_moulsot_feasibility_20260905.md).
The supervised live demo now uses local MoulSot; saved hosted settings remain
unchanged. See [live evidence](../../bench/results/local_voice_live_20260905.md).
The original BF16 assessment is retained below as historical evidence.

Local hosting may remove dependence on the shared Space queue, but neither a
successful model load nor a latency improvement has been demonstrated here.

The read-only hardware snapshot found an Intel i7-13620H (10 cores, 16 threads),
23.64 GiB of physical RAM with approximately 2.66 GiB free, and an NVIDIA RTX 4050
Laptop GPU with 6,141 MiB of VRAM. Approximately 4,088 MiB was occupied and
1,833 MiB free. These are transient availability measurements, not reservations.
Use nvidia-smi for VRAM capacity; the Windows CIM adapter value was misleading.

MoulSot's published BF16 weights are 4.08 GB (about 3.80 GiB), before activations
and runtime overhead. They do not fit the currently free VRAM. A short-turn,
batch-one service may be feasible with more free memory, but fit, Windows
compatibility, and speed remain unmeasured. No packages or models were installed
or downloaded for this check.

The GPU process inventory contained desktop, browser and game applications; no
project Python model worker appeared. Windows WDDM returned N/A for per-process
GPU memory, so the occupied memory cannot be attributed quantitatively to an
individual application. Do not stop the user's other applications to reclaim it.

If resources become available, use a separate environment and a loopback sidecar
through the existing MoulSot JSON adapter. Preserve the engine's torch 2.5.1 and
transformers 4.46.0 pins. Qwen3-ASR's published package requirements include
transformers 4.57.6 and accelerate 1.12.0; installing those into the engine
environment would change its tested dependency set. Start with the Transformers
backend; vLLM and FlashAttention are optional, not prerequisites for this check.

Before any local inference experiment, check free RAM/VRAM again and establish
an explicit memory/time bound. Report model load, peak memory, transcription,
and per-turn timing separately. Retain the actual MoulSot model and compare the
same saved payload without claiming synthetic input establishes native accuracy.

Primary references:

- [MoulSot model and model card](https://huggingface.co/atlasia/moulsot.v0.3)
- [Published model files](https://huggingface.co/atlasia/moulsot.v0.3/tree/main)
- [Qwen3-ASR inference backends](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-ASR package requirements](https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/main/pyproject.toml)
