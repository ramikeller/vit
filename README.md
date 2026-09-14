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
- Optimizer: Adam, learning rate `1e-3`, weight decay `1e-4`, batch size `32`.
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

The dataset is small (802 training images, 204 validation images,
evenly split between the two classes) and no random seed is fixed, so
a single run is not a reliable measure — weight initialization,
shuffling, and augmentation all vary run to run. Running the current
40-epoch configuration (dropout, weight decay, and horizontal-flip
augmentation all enabled) 5 times gives:

| | Final-epoch accuracy | Final-epoch loss | Best accuracy (any epoch) | Best loss (any epoch) |
| --- | --- | --- | --- | --- |
| Average | 65.1% | 0.624 | 69.8% | 0.604 |
| Range | 58.4% - 68.3% | 0.618 - 0.728 | 67.3% - 71.8% | 0.595 - 0.615 |

So: **the network reliably learns real signal well above the 50%
random-guess baseline, landing around two-thirds accuracy**, but it is
noisy from run to run and the gap between "final epoch" and "best
epoch reached" (65.1% vs 69.8% on average) shows the model doesn't
monotonically improve — it peaks partway through training and then
drifts. This is a modest result for cat-vs-dog classification (simple
CNNs or transfer learning on similar data typically reach the 90s%),
consistent with the very small, single-block architecture, low input
resolution, and small dataset used here.

This is the outcome of three rounds of tuning:

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
3. Adding horizontal-flip augmentation on top of that attacks the
   actual root cause — the dataset being small enough to memorize —
   rather than the symptom. The train/valid gap now stays much
   narrower throughout, and instead of collapsing by epoch 40,
   validation accuracy holds in the 60s-70s for most of the run (the
   averaged results above), rather than only in a narrow window around
   a single best epoch.

The model still peaks partway through training and then drifts a
little (the ~5-point gap between average final-epoch and average
best-epoch accuracy above is evidence of that), so it doesn't yet
squeeze out everything the current setup could give it. The codebase
does **not** currently implement early stopping, checkpointing, or
model saving — [`src/main.rs`](src/main.rs) always trains for the full
`NUM_EPOCHS`, the model is only ever held in memory and discarded when
the process exits, and the numbers above come from separately
re-running the program and recording the printed per-epoch output.
Adding early stopping (keep the best-validation-loss checkpoint),
more/varied augmentation, more training data, or a larger architecture
(more encoder blocks or attention heads) are the natural next steps,
in roughly that order of cost versus expected benefit — none of them
are implemented here yet.
