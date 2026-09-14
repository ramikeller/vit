use burn::{
    module::Module,
    nn::{
        LayerNorm, LayerNormConfig, Linear, LinearConfig,
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        transformer::{PositionWiseFeedForward, PositionWiseFeedForwardConfig},
    },
    tensor::{Device, Tensor, TensorData},
};

use crate::dataset::{NUM_PATCHES, PATCH_VALUES};

// Number of features carried by each patch token into the Transformer.
pub const EMBEDDING_SIZE: usize = 64;
// Frequency base for the fixed sine-cosine positional encoding.
pub const POSITION_BASE: f32 = 16.0;
pub const NUM_ATTENTION_HEADS: usize = 1;
pub const FFN_HIDDEN_SIZE: usize = 128;
pub const NUM_CLASSES: usize = 2;

#[derive(Debug, Clone, Copy)]
pub struct AttentionConfig {
    pub embedding_size: usize,
    pub num_heads: usize,
}

impl AttentionConfig {
    pub const fn new(embedding_size: usize, num_heads: usize) -> Self {
        assert!(num_heads > 0);
        assert!(embedding_size % num_heads == 0);

        Self {
            embedding_size,
            num_heads,
        }
    }
}

pub fn positional_encoding(device: &Device) -> Tensor<3> {
    let mut values = Vec::with_capacity(NUM_PATCHES * EMBEDDING_SIZE);
    let axis_width = EMBEDDING_SIZE / 2;

    for row in 0..4 {
        for column in 0..4 {
            for position in [row, column] {
                for frequency in 0..axis_width / 2 {
                    let angle = position as f32
                        / POSITION_BASE.powf((2 * frequency) as f32 / axis_width as f32);
                    values.push(angle.sin());
                    values.push(angle.cos());
                }
            }
        }
    }

    Tensor::from_data(
        TensorData::new(values, [1, NUM_PATCHES, EMBEDDING_SIZE]),
        device,
    )
}

#[derive(Module, Debug)]
pub struct PatchEmbedding {
    projection: Linear,
}

impl PatchEmbedding {
    pub fn new(device: &Device) -> Self {
        Self {
            projection: LinearConfig::new(PATCH_VALUES, EMBEDDING_SIZE).init(device),
        }
    }

    pub fn forward(&self, patches: Tensor<3>) -> Tensor<3> {
        let [batch_size, patch_count, patch_values] = patches.dims();
        let device = patches.device();
        assert_eq!(
            patch_count, NUM_PATCHES,
            "unexpected number of image patches"
        );
        assert_eq!(patch_values, PATCH_VALUES, "unexpected patch feature count");

        self.projection
            .forward(patches)
            .reshape([batch_size, NUM_PATCHES, EMBEDDING_SIZE])
            .add(positional_encoding(&device))
    }
}

#[derive(Module, Debug)]
pub struct SelfAttention {
    attention: MultiHeadAttention,
}

impl SelfAttention {
    pub fn new(config: AttentionConfig, device: &Device) -> Self {
        Self {
            attention: MultiHeadAttentionConfig::new(config.embedding_size, config.num_heads)
                .with_dropout(0.0)
                .init(device),
        }
    }

    pub fn forward(&self, tokens: Tensor<3>) -> Tensor<3> {
        self.attention.forward(MhaInput::self_attn(tokens)).context
    }
}

#[derive(Module, Debug)]
pub struct FeedForward {
    network: PositionWiseFeedForward,
}

impl FeedForward {
    pub fn new(embedding_size: usize, hidden_size: usize, device: &Device) -> Self {
        Self {
            network: PositionWiseFeedForwardConfig::new(embedding_size, hidden_size)
                .with_dropout(0.0)
                .init(device),
        }
    }

    pub fn forward(&self, tokens: Tensor<3>) -> Tensor<3> {
        self.network.forward(tokens)
    }
}

#[derive(Module, Debug)]
pub struct TransformerBlock {
    attention_norm: LayerNorm,
    attention: SelfAttention,
    feed_forward_norm: LayerNorm,
    feed_forward: FeedForward,
}

impl TransformerBlock {
    pub fn new(config: AttentionConfig, hidden_size: usize, device: &Device) -> Self {
        Self {
            attention_norm: LayerNormConfig::new(config.embedding_size).init(device),
            attention: SelfAttention::new(config, device),
            feed_forward_norm: LayerNormConfig::new(config.embedding_size).init(device),
            feed_forward: FeedForward::new(config.embedding_size, hidden_size, device),
        }
    }

    pub fn forward(&self, tokens: Tensor<3>) -> Tensor<3> {
        let attention_input = self.attention_norm.forward(tokens.clone());
        let tokens = tokens.add(self.attention.forward(attention_input));
        let feed_forward_input = self.feed_forward_norm.forward(tokens.clone());

        tokens.add(self.feed_forward.forward(feed_forward_input))
    }
}

pub fn mean_pool_tokens(tokens: Tensor<3>) -> Tensor<2> {
    let [batch_size, patch_count, embedding_size] = tokens.dims();
    assert_eq!(
        patch_count, NUM_PATCHES,
        "unexpected number of image patches"
    );
    assert_eq!(
        embedding_size, EMBEDDING_SIZE,
        "unexpected token feature count"
    );

    tokens.mean_dim(1).reshape([batch_size, EMBEDDING_SIZE])
}

#[derive(Module, Debug)]
pub struct ClassificationHead {
    projection: Linear,
}

impl ClassificationHead {
    pub fn new(device: &Device) -> Self {
        Self {
            projection: LinearConfig::new(EMBEDDING_SIZE, NUM_CLASSES).init(device),
        }
    }

    pub fn forward(&self, pooled_tokens: Tensor<2>) -> Tensor<2> {
        let [batch_size, embedding_size] = pooled_tokens.dims();
        assert_eq!(
            embedding_size, EMBEDDING_SIZE,
            "unexpected pooled feature count"
        );

        self.projection
            .forward(pooled_tokens)
            .reshape([batch_size, NUM_CLASSES])
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Tensor, TensorData};

    use super::{
        AttentionConfig, ClassificationHead, EMBEDDING_SIZE, FFN_HIDDEN_SIZE, FeedForward,
        NUM_ATTENTION_HEADS, NUM_CLASSES, PatchEmbedding, SelfAttention, TransformerBlock,
        mean_pool_tokens, positional_encoding,
    };
    use crate::dataset::{NUM_PATCHES, PATCH_VALUES};

    #[test]
    fn patch_embedding_projects_patches_to_model_width() {
        let device = Default::default();
        let patches = Tensor::from_data(
            TensorData::new(
                vec![0.0; 2 * NUM_PATCHES * PATCH_VALUES],
                [2, NUM_PATCHES, PATCH_VALUES],
            ),
            &device,
        );
        let embedding = PatchEmbedding::new(&device);

        let embedded = embedding.forward(patches);

        assert_eq!(embedded.dims(), [2, NUM_PATCHES, EMBEDDING_SIZE]);
    }

    #[test]
    fn positional_encoding_has_expected_shape_and_origin() {
        let device = Default::default();
        let encoding = positional_encoding(&device);
        let values = encoding.to_data().as_slice::<f32>().unwrap().to_vec();

        assert_eq!(encoding.dims(), [1, NUM_PATCHES, EMBEDDING_SIZE]);
        assert_eq!(values[0], 0.0);
        assert_eq!(values[1], 1.0);
        assert_eq!(values[32], 0.0);
        assert_eq!(values[33], 1.0);

        let column_one_offset = EMBEDDING_SIZE;
        assert!((values[column_one_offset + 32] - 1.0_f32.sin()).abs() < 1e-6);
        assert!((values[column_one_offset + 33] - 1.0_f32.cos()).abs() < 1e-6);
    }

    #[test]
    fn single_head_attention_preserves_token_shape() {
        let device = Default::default();
        let tokens = Tensor::from_data(
            TensorData::new(
                vec![0.0; 2 * NUM_PATCHES * EMBEDDING_SIZE],
                [2, NUM_PATCHES, EMBEDDING_SIZE],
            ),
            &device,
        );
        let attention = SelfAttention::new(
            AttentionConfig::new(EMBEDDING_SIZE, NUM_ATTENTION_HEADS),
            &device,
        );

        let output = attention.forward(tokens);

        assert_eq!(output.dims(), [2, NUM_PATCHES, EMBEDDING_SIZE]);
    }

    #[test]
    fn feed_forward_preserves_token_shape() {
        let device = Default::default();
        let tokens = Tensor::from_data(
            TensorData::new(
                vec![0.0; 2 * NUM_PATCHES * EMBEDDING_SIZE],
                [2, NUM_PATCHES, EMBEDDING_SIZE],
            ),
            &device,
        );
        let feed_forward = FeedForward::new(EMBEDDING_SIZE, FFN_HIDDEN_SIZE, &device);

        let output = feed_forward.forward(tokens);

        assert_eq!(output.dims(), [2, NUM_PATCHES, EMBEDDING_SIZE]);
    }

    #[test]
    fn transformer_block_preserves_token_shape() {
        let device = Default::default();
        let tokens = Tensor::from_data(
            TensorData::new(
                vec![0.0; 2 * NUM_PATCHES * EMBEDDING_SIZE],
                [2, NUM_PATCHES, EMBEDDING_SIZE],
            ),
            &device,
        );
        let block = TransformerBlock::new(
            AttentionConfig::new(EMBEDDING_SIZE, NUM_ATTENTION_HEADS),
            FFN_HIDDEN_SIZE,
            &device,
        );

        let output = block.forward(tokens);

        assert_eq!(output.dims(), [2, NUM_PATCHES, EMBEDDING_SIZE]);
    }

    #[test]
    fn mean_pooling_and_classification_produce_logits() {
        let device = Default::default();
        let tokens = Tensor::from_data(
            TensorData::new(
                vec![0.0; 2 * NUM_PATCHES * EMBEDDING_SIZE],
                [2, NUM_PATCHES, EMBEDDING_SIZE],
            ),
            &device,
        );
        let pooled = mean_pool_tokens(tokens);
        let classifier = ClassificationHead::new(&device);

        let logits = classifier.forward(pooled);

        assert_eq!(logits.dims(), [2, NUM_CLASSES]);
    }
}
