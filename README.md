# Efficient Self Attention for Perceptual Similarity

Master's thesis project on efficient self-attention mechanisms for perceptual image similarity (MSc in Computer Engineering, Sapienza University of Rome).

## Project Goals

- Evaluate how efficient attention mechanisms can replace MHA in a transformer backbone for perceptual similarity.
- Measure the trade-off between efficiency gains and accuracy loss for each efficient variant.
- Compare theoretical efficiency gains with empirical runtime improvements.
- Show how knowledge distillation can recover a large portion of the lost accuracy.

## Resources and Credits

- NIGHTS dataset info and download instructions are available in the [DreamSim repository](https://github.com/ssundaram21/dreamsim/tree/main/dataset).
- ImageNet-100 dataset info and download instructions are available on [Hugging Face](https://huggingface.co/datasets/clane9/imagenet-100).
- Part of this codebase is adapted from [DreamSim](https://github.com/ssundaram21/dreamsim/tree/main).
- Efficient attention modules in this repository are implemented from scratch and may differ from the original papers: [SRA](https://arxiv.org/abs/2102.12122), [Pool](https://arxiv.org/abs/2111.11418), [MoH](https://arxiv.org/abs/2410.11842), [SOFT](https://arxiv.org/abs/2110.11945).

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

**Note:** After the first model run (which downloads DINO weights to `models/`), update one import in the downloaded file. Edit `models/facebookresearch_dino_main/vision_transformer.py` (line 24):

```python
# Change from:
from utils import trunc_normal_

# To:
from timm.layers import trunc_normal_
```

## Evaluation

Run 2AFC evaluation on NIGHTS test split:

```bash
python evaluation.py --dataset_root ./nights --split test --device cuda --attention_module benchmark
```

Minimal toy example (CPU, runs in seconds):

```bash
python evaluation.py --dataset_root ./nights --split test --device cpu --max_samples 16 --max_batches 1 --batch_size 4 --warmup_batches 1
```

## Embedding Extraction

Extract embeddings on NIGHTS:

```bash
python -m training.embedding --dataset_root ./nights --split all --device cuda --output_path ./training/embeddings/nights.pt
```

## Knowledge Distillation

Distill the pool attention model using teacher embeddings from the benchmark (LoRA-finetuned DINO) model:

```bash
python -m training.distill_pool --teacher_embeddings ./training/embeddings/nights.pt --train_split train --val_split val --epochs 50 --batch_size 32 --device cuda --eval_2afc_every 25 --eval_2afc_split val
```