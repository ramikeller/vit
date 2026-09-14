# vit

A small Vision Transformer, implemented from scratch in Rust on top of the
[`burn`](https://burn.dev) deep learning framework, trained to classify
images as cats or dogs.

## Components

| File | Responsibility |
| --- | --- |
| [`src/dataset.rs`](src/dataset.rs) | Discovers labeled image samples from `data/train` and `data/valid`, decodes and resizes them into normalized `[C, H, W]` tensors (optionally applying data augmentation), batches samples into `[N, C, H, W]` tensors with aligned labels, and patchifies images into flattened non-overlapping patches for the transformer. |
| [`src/model.rs`](src/model.rs) | Defines the model: fixed sine-cosine positional encoding, patch embedding (linear projection of patches to the embedding size), a single-head self-attention block, a position-wise feed-forward network, a pre-norm Transformer encoder block, mean pooling over patch tokens, a linear classification head, and the `VisionTransformer` that composes them end to end. Attention and the feed-forward network apply dropout during training. Also defines the cross-entropy classification loss. |
| [`src/training.rs`](src/training.rs) | Training and evaluation logic: a single gradient-descent step (`train_step`) using the Adam optimizer (with weight decay), a full-epoch training loop over batches of samples (`train_epoch`), and validation (`evaluate`) that runs the model in eval mode (dropout disabled) and reports loss and classification accuracy. |
| [`src/main.rs`](src/main.rs) | Entry point: discovers the train/validation splits, then runs the training loop for a fixed number of epochs, shuffling the training samples each epoch and printing training loss, validation loss, and validation accuracy. |

### Architecture

- Input images: 32x32 RGB, split into 8x8 patches (16 patches per image, 192 raw values per patch).
- Embedding size: 64.
- Positional encoding: fixed sine-cosine, added to patch embeddings.
- Encoder: 1 Transformer block (pre-norm), 1 attention head, feed-forward hidden size 128.
- Pooling: mean over the 16 patch tokens.
- Classification head: linear projection to 2 classes (`cats`, `dogs`).
- Loss: cross-entropy.
- Optimizer: Adam, learning rate `1e-3`, weight decay `1e-4`.
- Regularization: dropout `0.1` in attention and the feed-forward network (active only during training; validation runs the model in eval mode with dropout disabled).
- Data augmentation: each training image is randomly flipped horizontally (50% chance), re-decoded fresh every epoch; validation images are never augmented.

This is intentionally a minimal, single-block ViT rather than a
state-of-the-art architecture — the project's focus is building the
Transformer machinery from first principles.

## Running

```sh
cargo test           # unit tests for each component
cargo run --release  # train and validate for 40 epochs
```

Training data is expected under `data/train/{cats,dogs}/` and validation
data under `data/valid/{cats,dogs}/`, as JPEG or PNG files.

## Training and validation results

A 40-epoch run over the current dataset (802 training images, 204
validation images, evenly split between the two classes), with dropout,
weight decay, and horizontal-flip augmentation all enabled:

| Epoch | Train loss | Valid loss | Valid accuracy |
| --- | --- | --- | --- |
| 1 | 0.7454 | 0.6885 | 50.5% |
| 5 | 0.6993 | 0.6803 | 62.4% |
| 10 | 0.6742 | 0.6266 | 68.3% |
| 16 | 0.6647 | 0.6222 | 71.3% (best accuracy) |
| 21 | 0.6510 | 0.6128 | 71.3% |
| 29 | 0.6309 | **0.6075** (best loss) | 68.3% |
| 40 | 0.6038 | 0.6499 | 65.8% |

This history reflects three rounds of tuning:

1. An earlier 5-epoch run (before any regularization) barely moved off
   the 50% random-guess baseline — that pointed to underfitting, not
   overfitting, so the first fix was simply training longer.
2. Bumping to 40 epochs without regularization showed the real shape of
   the problem: validation loss dropped sharply through roughly epoch
   15-17 (down to ~0.60), then climbed back up past epoch 20 while
   training loss kept falling — classic overfitting on this small,
   ~800-image dataset. Adding dropout (`0.1`) and Adam weight decay
   (`1e-4`) delayed this (the good region stretched to roughly epochs
   10-30) but didn't eliminate it — by epoch 40 validation loss had
   climbed back to 0.73. (A much stronger weight decay of `1e-2` was
   also tried and made things worse — the model could barely learn at
   all, since it's tiny and doesn't have capacity to spare.)
3. Adding horizontal-flip augmentation on top of that (the results
   above) attacks the actual root cause — the dataset being small
   enough to memorize — rather than the symptom. The train/valid gap
   now stays much narrower throughout, and instead of collapsing by
   epoch 40, validation accuracy holds in the 60s-70s for the whole
   run. Best validation loss (0.6075) is about the same as before, but
   it's sustained over a much wider window (roughly epochs 20-40
   instead of a narrow peak around epoch 18), and the model no longer
   needs to be caught at just the right epoch to get a good result.

The remaining gap between train and validation loss is still there,
which means more training data, more augmentation (random crops, color
jitter), or a stronger architecture (more encoder blocks/heads) would
all plausibly push accuracy higher from here. **Early stopping** —
tracking the best validation loss and keeping that checkpoint rather
than the final epoch's weights — remains a cheap, safe addition
regardless of any of the above.
