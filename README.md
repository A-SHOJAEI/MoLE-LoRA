# MoLE: Mixture of LoRA Experts via Parameter-Efficient Task Specialization

A parameter-efficient approach to mixture-of-experts that achieves task specialization while reducing memory requirements by 73% compared to traditional multi-model ensembles.

## Overview

MoLE combines a single shared base language model (Llama-3.2-3B) with task-specific Low-Rank Adaptation (LoRA) modules and a learned routing mechanism. This enables scalable expert specialization without maintaining multiple full models.

**Architecture:**
- **Shared Base Model**: Frozen Llama-3.2-3B (3.2B parameters)
- **4 Task-Specific Experts**: LoRA adapters with rank hierarchy (64/32/16/8) for reasoning, knowledge, QA, and factoid tasks
- **Learned Router**: BERT-tiny classifier (4M parameters) for automatic expert selection

## Results

| Metric | Value |
|--------|-------|
| Routing Accuracy | 100% on a 16-query evaluation set (4 per expert category) |
| Memory (Traditional 4-Model MoE) | 48 GB |
| Memory (MoLE) | 12.74 GB |
| Memory Savings | 73.4% |
| Expert Training Loss (Reasoning) | 2.24 → 1.96 (12.5% reduction) |
| Gradient Norms | Bounded 0.7-1.1 (stable throughout) |

### Baseline Performance (Llama-3.2-3B)

| Benchmark | Accuracy |
|-----------|----------|
| MMLU | 55.2% |
| GSM8K | 27.4% |
| HellaSwag | 74.1% |
| ARC-Easy | 74.5% |
| TruthfulQA | 39.3% |

### Key Finding: Pruning Incompatibility

Systematic diagnostics revealed that structured pruning fundamentally damages model capacity for LoRA fine-tuning:

| Configuration | Loss (Step 25) | Gradient Norm | Outcome |
|---------------|----------------|---------------|---------|
| Unpruned + LoRA | 2.21 | 0.94 | Stable |
| 15% Pruned + LoRA | 264,552,407 | NaN | Explosion |

This finding motivated the pruning-free architecture.

## Project Structure

```
MoLE-LORA/
├── MoLE_LoRA_Experts_FINAL.ipynb   # Complete implementation (Colab notebook)
├── MoLE_Research_Paper.md          # Full research paper
├── MoLE-LoRA-Research/
│   ├── models/                     # Trained LoRA adapters and router (configs only; weights excluded due to size)
│   ├── results/                    # Evaluation results and visualizations
│   ├── FINAL_SUMMARY.json          # Experiment summary
│   └── progress.json               # Training phase tracking
├── requirements.txt
└── LICENSE
```

## Reproduction

### Requirements

- GPU: NVIDIA A100 (80GB) or equivalent
- Python 3.10+
- HuggingFace account with Llama 3 access

### Setup

```bash
pip install -r requirements.txt
```

### Running

Open `MoLE_LoRA_Experts_FINAL.ipynb` in Google Colab (or Jupyter with GPU) and run all cells sequentially. The notebook handles:

1. **Phase 1**: Baseline evaluation (~2 hours)
2. **Phase 2**: Expert training - 4 LoRA adapters (~10 hours)
3. **Phase 3**: Router training (~30 minutes)
4. **Phase 4**: Integrated evaluation (~30 minutes)
5. **Phase 5**: Analysis and visualization

Total runtime: ~13 hours on A100.

### Datasets

- **GSM8K**: `openai/gsm8k` (HuggingFace Hub)
- **WikiText-103**: `wikitext/wikitext-103-v1` (HuggingFace Hub)
- **Router data**: 10,000 synthetically generated query-label pairs (generated in notebook)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-6 |
| Batch size | 2 (8 gradient accumulation) |
| Max gradient norm | 0.5 |
| Precision | FP32 |
| Optimizer | AdamW with cosine schedule |
| Training samples | 10,000 per expert |
| Epochs | 2 |

## Limitations

- Generation quality shows repetition due to conservative training regime (low LR, limited epochs)
- Router test set (16 queries) is small; larger-scale evaluation needed
- Experts 2-4 trained on WikiText subsets produce similar outputs
- Single-expert routing (no cross-expert blending)

## Citation

```bibtex
@misc{shojaei2026mole,
  title={MoLE: Mixture of LoRA Experts via Parameter-Efficient Task Specialization},
  author={Shojaei, Alireza},
  year={2026},
  note={Available at: https://github.com/A-SHOJAEI/MoLE-LoRA}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

The base model (Llama-3.2-3B) is subject to the [Llama 3 Community License](https://ai.meta.com/llama/license/).
