use burn::{
    optim::{AdamConfig, GradientsParams, ModuleOptimizer},
    tensor::{Int, Tensor},
};

use crate::model::{VisionTransformer, classification_loss};

pub const LEARNING_RATE: f64 = 1.0e-3;

pub fn new_adam() -> ModuleOptimizer {
    AdamConfig::new().init()
}

pub fn train_step(
    model: VisionTransformer,
    optimizer: &mut ModuleOptimizer,
    images: Tensor<4>,
    labels: Tensor<1, Int>,
) -> (VisionTransformer, f32) {
    let logits = model.forward(images);
    let loss = classification_loss(logits, labels);
    let loss_value = loss.to_data().as_slice::<f32>().unwrap()[0];
    let gradients = GradientsParams::from_grads(loss.backward(), &model);
    let updated_model = optimizer.step(LEARNING_RATE, model, gradients);

    (updated_model, loss_value)
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Device, Tensor, TensorData};

    use super::{LEARNING_RATE, new_adam, train_step};
    use crate::model::{NUM_CLASSES, VisionTransformer};

    #[test]
    fn one_training_step_returns_finite_loss() {
        let device = Device::default().autodiff();
        let model = VisionTransformer::new(&device);
        let mut optimizer = new_adam();
        let images = Tensor::from_data(
            TensorData::new(vec![0.0; 2 * 3 * 32 * 32], [2, 3, 32, 32]),
            &device,
        );
        let labels = Tensor::from_data(TensorData::new(vec![0i64, 1], [2]), &device);

        let (_, loss) = train_step(model, &mut optimizer, images, labels);

        assert!(loss.is_finite());
        assert!(LEARNING_RATE > 0.0);
        assert_eq!(NUM_CLASSES, 2);
    }
}
