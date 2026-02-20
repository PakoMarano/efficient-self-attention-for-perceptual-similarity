# Efficient Self Attention for Perceptual Similarity
Master thesis work on efficient self attention mechanisms for computing perceptual similarity between images (MSc in Computer Engineering Sapienza)

Informations about the NIGHTS dataset and instructions on how to download it can be found in the [Dreamsim repository](https://github.com/ssundaram21/dreamsim/tree/main/dataset).
Part of the code has been adapted from the code in the [Dreamsim repository](https://github.com/ssundaram21/dreamsim/tree/main).

## Evaluation

Run 2AFC evaluation on NIGHTS test split:

```bash
python evaluation.py --dataset_root ./nights --split test --device cuda
```

Quick subset evaluation (for fast checks):

```bash
python evaluation.py --dataset_root ./nights --split test --max_samples 128 --max_batches 8
```

Notes:
- `--impl` is accepted and logged, but is currently a no-op.
- Results are appended to `./reports/results.csv` by default.