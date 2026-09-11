use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Device, Tensor, TensorData},
};

use crate::dataset::{NUM_PATCHES, PATCH_VALUES};

pub const EMBEDDING_SIZE: usize = 64;
pub const POSITION_BASE: f32 = 16.0;

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

#[cfg(test)]
mod tests {
    use burn::tensor::{Tensor, TensorData};

    use super::{EMBEDDING_SIZE, PatchEmbedding, positional_encoding};
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
}
