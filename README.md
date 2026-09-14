# vit

A small Vision Transformer, implemented from scratch in Rust on top of the
[`burn`](https://burn.dev) deep learning framework, trained to classify
images as cats or dogs.

## Components

| File | Responsibility |
| --- | --- |
| [`src/dataset.rs`](src/dataset.rs) | Discovers labeled image samples from `data/train` and `data/valid`, decodes and resizes them into normalized `[C, H, W]` tensors, batches samples into `[N, C, H, W]` tensors with aligned labels, and patchifies images into flattened non-overlapping patches for the transformer. |
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
validation images, evenly split between the two classes):

| Epoch | Train loss | Valid loss | Valid accuracy |
| --- | --- | --- | --- |
| 1 | 0.7316 | 0.6838 | 54.0% |
| 5 | 0.6848 | 0.6433 | 63.9% |
| 10 | 0.6602 | 0.6187 | 65.3% |
| 18 | 0.6310 | **0.6094** (best) | 69.3% |
| 25 | 0.6244 | 0.6222 | 65.8% |
| 32 | 0.5891 | 0.6670 | 64.9% |
| 40 | 0.5331 | 0.7326 | 62.9% |

An earlier 5-epoch run (before regularization was added) barely moved
off the 50% random-guess baseline — that pointed to underfitting, not
overfitting, so the first fix was simply training longer. Bumping to 40
epochs (still without regularization) showed the real shape of the
problem: validation loss dropped sharply through roughly epoch 15-17
(down to ~0.60), then climbed back up past epoch 20 while training loss
kept falling — classic overfitting on this small, ~800-image dataset.

Adding dropout (`0.1`) and a small amount of Adam weight decay didn't
eliminate overfitting, but it did help: the best validation loss improved
slightly and the "good" region of high validation accuracy (roughly
mid-60s to 69%) stretched from about 10 epochs (7-17) to about 20 epochs
(10-30) before degrading. A much stronger weight decay (`1e-2`) was also
tried and made things worse — the model could barely learn at all,
staying near the random-guess loss for the full 40 epochs, since the
model is tiny and doesn't have much capacity to spare.

Since overfitting still eventually sets in — an artifact of the model
continuing to train well past its useful point on a small dataset,
not of insufficient regularization — the clearest next step is **early
stopping**: track the best validation loss during training and keep that
checkpoint instead of the final-epoch weights. Beyond that, more training
data or augmentation would attack the root cause (dataset size) rather
than treating the symptom.
