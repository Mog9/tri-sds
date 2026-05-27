# tri-sds — Triton Speculative Decoding Inference
## What is Speculative Decoding?
SD speeds up LLM inference by using a small draft model to propose K tokens, then verifying them in a single target forward pass. If the target agrees, you get K tokens for the cost of 1 target step (plus cheap draft steps). Acceptance rates of 0.8-0.95 are common when draft and target share a family.

<img width="800" height="910" alt="sds" src="https://github.com/user-attachments/assets/5e95d939-b175-4e2f-9d97-ce1ed96ad0ee" />

---
## What this project is
A custom speculative decoding engine built in Triton, designed as a drop-in plugin for SGLang (v0.5.11). The goal is to provide an alternative SD backend with correct-by-construction verify logic and Triton kernels for GQA decode attention, targeting the SGLang Docker runtime on AMD MI300X.

---
## Correctness Benchmarks
Validated the full pipeline (prefill, decode, and spec verify) against vanilla PyTorch/HF — 100% argmax match across all modes. I've tested with the GPT family because thats what i can run on my personal GPU (4gb) and I'm outperforming SGLang on both throughput and correctness on this task. but ik SGLang isn't really tuned for this gap and the model is too small for SD to properly work. i am benchmarking bigger models

<img width="1170" height="645" alt="my gpu correctness" src="https://github.com/user-attachments/assets/1beed17a-03e1-4785-a26d-bfac7ba36b31" />

---
## Qwen3 on AMD MI300X
Currently benchmarking Qwen3-4B (draft) → Qwen3-32B (target) with Triton GQA decode kernels on a single AMD MI300X. results coming soon.