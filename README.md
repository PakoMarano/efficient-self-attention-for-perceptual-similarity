# Efficient Self Attention for Perceptual Similarity

Master's thesis project on efficient self-attention mechanisms for perceptual image similarity (MSc in Computer Engineering, Sapienza University of Rome).

## Project Goals

- Evaluate how efficient attention mechanisms can replace MHA in a transformer backbone for perceptual similarity.
- Measure the trade-off between efficiency gains and accuracy loss for each efficient variant.
- Compare theoretical efficiency gains with empirical runtime improvements.
- Show how knowledge distillation on a MetaFormer-based architecture can recover a large portion of the lost accuracy.
- Propose a model that gets XX 2AFC accuracy on NIGHTS and is X times faster than DreamSim.

## Resources and Credits

- NIGHTS dataset info and download instructions are available in the [DreamSim repository](https://github.com/ssundaram21/dreamsim/tree/main/dataset).
- ImageNet-100 dataset info and download instructions are available on [Hugging Face](https://huggingface.co/datasets/ilee0022/ImageNet100).
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
python evaluation.py --dataset_root ./nights --split test --device cuda --attention_module mha
```

Minimal toy example (CPU, runs in seconds):

```bash
python evaluation.py --dataset_root ./nights --split test --device cpu --max_samples 16 --max_batches 1 --batch_size 4 --warmup_batches 1
```

## Embedding Extraction

If you downloaded ImageNet100 from Hugging Face (parquet shards), convert it once to image folders:

```bash
python ./scripts/convert_imagenet100_from_parquet.py --dataset_root ./imagenet-100 --output_root ./imagenet-100-images --splits train validation test
```

Extract embeddings on NIGHTS:

```bash
python -m training.embedding --dataset_root ./nights --device cuda --output_path ./training/embeddings/nights.pt
```

Extract embeddings on NIGHTS + ImageNet100:

```bash
python -m training.embedding --dataset_root ./nights --device cuda --extra_image_roots ./imagenet-100-images --output_path ./training/embeddings/nights_imagenet100.pt
```

## Knowledge Distillation

Distill the pool attention model using teacher embeddings from the MHA (LoRA-finetuned DINO) model:

```bash
python -m training.distill_pool --teacher_embeddings ./training/embeddings/nights.pt --train_split train --val_split val --epochs 50 --batch_size 32 --device cuda --eval_2afc_every 25 --eval_2afc_split val
```

Distill with extra ImageNet100 images in train split (val remains NIGHTS-only):

```bash
python -m training.distill_pool --teacher_embeddings ./training/embeddings/nights_imagenet100.pt --train_split train --val_split val --epochs 75 --batch_size 128 --num_workers 2 --device cuda --extra_image_roots ./imagenet-100-images --eval_2afc_every 15 --eval_2afc_split val
```