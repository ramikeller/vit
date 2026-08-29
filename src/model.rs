use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Device, Tensor},
};

use crate::dataset::{NUM_PATCHES, PATCH_VALUES};

pub const EMBEDDING_SIZE: usize = 64;

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
        assert_eq!(
            patch_count, NUM_PATCHES,
            "unexpected number of image patches"
        );
        assert_eq!(patch_values, PATCH_VALUES, "unexpected patch feature count");

        self.projection
            .forward(patches)
            .reshape([batch_size, NUM_PATCHES, EMBEDDING_SIZE])
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Tensor, TensorData};

    use super::{EMBEDDING_SIZE, PatchEmbedding};
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
}
