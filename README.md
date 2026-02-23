# Efficient Self Attention for Perceptual Similarity
Master thesis work on efficient self attention mechanisms for computing perceptual similarity between images (MSc in Computer Engineering Sapienza)

- Informations about the NIGHTS dataset can be found in the [Dreamsim repository](https://github.com/ssundaram21/dreamsim/tree/main/dataset).
- Part of the code has been adapted from the code in the [Dreamsim repository](https://github.com/ssundaram21/dreamsim/tree/main).
- The code for efficient modules has been implemented from scratch and thus maybe very different than the original implemetations. Anyway, these are the papers in which the original attention models have been introduced: [SRA](https://arxiv.org/abs/2102.12122), [Pool](https://arxiv.org/abs/2111.11418), [MoH](https://arxiv.org/abs/2410.11842), [SOFT](https://arxiv.org/abs/2110.11945).

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

**Note:** After the first model run (which downloads DINO weights to `models/`), you'll need to fix an import in the downloaded file. Edit `models/facebookresearch_dino_main/vision_transformer.py` line 24:

```python
# Change from:
from utils import trunc_normal_

# To:
from timm.layers import trunc_normal_
```

## Evaluation

Run 2AFC evaluation on NIGHTS test split:

```bash
python evaluation.py --dataset_root ./nights --split test --device cuda
```

Minimal toy example (CPU, runs in seconds):

```bash
python evaluation.py --dataset_root ./nights --split test --device cpu --max_samples 16 --max_batches 1 --batch_size 4 --warmup_batches 1
```

Notes:
- `--attention_module` selects the ViT attention backend; `benchmark` keeps standard multi-head attention.
- Results are appended to `./reports/results.csv` by default.

## Embedding Extraction

Run embedding extraction on NIGHTS:

```bash
python -m training.embedding --dataset_root ./nights --split all --device cuda --output_path ./training/embeddings/nights.pt
```

Minimal toy example (CPU, runs in seconds):

```bash
python -m training.embedding --dataset_root ./nights --split test --device cpu --max_samples 16 --max_batches 1 --batch_size 4 --warmup_batches 1 --output_path ./training/embeddings/nights_toy.pt
```

Notes:
- Embeddings are saved to `--output_path` as a `.pt` file with tensors and run metadata.
- Run logs are appended to `./reports/embedding_runs.csv` by default.