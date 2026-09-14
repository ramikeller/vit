# vit

A small Vision Transformer, implemented from scratch in Rust on top of the
[`burn`](https://burn.dev) deep learning framework, trained to classify
images as cats or dogs.

## Components

| File | Responsibility |
| --- | --- |
| [`src/dataset.rs`](src/dataset.rs) | Discovers labeled image samples from `data/train` and `data/valid`, decodes and resizes them into normalized `[C, H, W]` tensors, batches samples into `[N, C, H, W]` tensors with aligned labels, and patchifies images into flattened non-overlapping patches for the transformer. |
| [`src/model.rs`](src/model.rs) | Defines the model: fixed sine-cosine positional encoding, patch embedding (linear projection of patches to the embedding size), a single-head self-attention block, a position-wise feed-forward network, a pre-norm Transformer encoder block, mean pooling over patch tokens, a linear classification head, and the `VisionTransformer` that composes them end to end. Also defines the cross-entropy classification loss. |
| [`src/training.rs`](src/training.rs) | Training and evaluation logic: a single gradient-descent step (`train_step`) using the Adam optimizer, a full-epoch training loop over batches of samples (`train_epoch`), and validation (`evaluate`) that reports loss and classification accuracy. |
| [`src/main.rs`](src/main.rs) | Entry point: discovers the train/validation splits, then runs the training loop for a fixed number of epochs, shuffling the training samples each epoch and printing training loss, validation loss, and validation accuracy. |

### Architecture

- Input images: 32x32 RGB, split into 8x8 patches (16 patches per image, 192 raw values per patch).
- Embedding size: 64.
- Positional encoding: fixed sine-cosine, added to patch embeddings.
- Encoder: 1 Transformer block (pre-norm), 1 attention head, feed-forward hidden size 128.
- Pooling: mean over the 16 patch tokens.
- Classification head: linear projection to 2 classes (`cats`, `dogs`).
- Loss: cross-entropy.
- Optimizer: Adam, learning rate `1e-3`.

This is intentionally a minimal, single-block ViT rather than a
state-of-the-art architecture — the project's focus is building the
Transformer machinery from first principles.

## Running

```sh
cargo test           # unit tests for each component
cargo run --release  # train and validate for 5 epochs
```

Training data is expected under `data/train/{cats,dogs}/` and validation
data under `data/valid/{cats,dogs}/`, as JPEG or PNG files.

## Training and validation results

A 5-epoch run over the current dataset (802 training images, 204
validation images, evenly split between the two classes):

| Epoch | Train loss | Valid loss | Valid accuracy |
| --- | --- | --- | --- |
| 1 | 0.7091 | 0.7135 | 50.0% |
| 2 | 0.6906 | 0.7193 | 50.0% |
| 3 | 0.6909 | 0.6845 | 53.0% |
| 4 | 0.6885 | 0.6563 | 58.4% |
| 5 | 0.6850 | 0.6872 | 51.5% |

Training loss decreases steadily, but validation accuracy stays close to
the 50% random-guess baseline for this task, with a lot of run-to-run
variance (weights are randomly initialized and training samples are
reshuffled each epoch, with no fixed seed). This is expected for a
single-block, single-head, 32x32-input ViT trained without data
augmentation or regularization — the model has very limited capacity.
Increasing the number of encoder blocks/attention heads, training for
more epochs, and adding augmentation would be the natural next steps to
improve accuracy.
