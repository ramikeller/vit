use burn::{
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, ModuleOptimizer, decay::WeightDecayConfig},
    tensor::{Device, Int, Tensor},
};

use crate::dataset::{Batch, Sample, load_batch};
use crate::model::{VisionTransformer, classification_loss};

pub const LEARNING_RATE: f64 = 1.0e-3;
pub const BATCH_SIZE: usize = 32;
// L2 penalty applied to weights by the Adam optimizer, to discourage overfitting.
pub const WEIGHT_DECAY: f32 = 1.0e-4;

pub fn new_adam() -> ModuleOptimizer {
    AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(WEIGHT_DECAY)))
        .init()
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

pub fn train_epoch(
    model: VisionTransformer,
    optimizer: &mut ModuleOptimizer,
    samples: &[Sample],
    device: &Device,
) -> (VisionTransformer, f32) {
    let mut model = model;
    let mut total_loss = 0.0;
    let mut num_batches = 0;

    for batch in samples.chunks(BATCH_SIZE) {
        let Batch { images, labels } =
            load_batch(batch, device, true).expect("failed to load training batch");
        let (updated_model, loss) = train_step(model, optimizer, images, labels);
        model = updated_model;
        total_loss += loss;
        num_batches += 1;
    }

    (model, total_loss / num_batches as f32)
}

pub fn evaluate(model: &VisionTransformer, samples: &[Sample], device: &Device) -> (f32, f32) {
    // Evaluate on the inner (non-autodiff) device so dropout is disabled, matching train/eval mode.
    let eval_device = device.clone().inner();
    let eval_model = model.valid();
    let mut total_loss = 0.0;
    let mut correct = 0.0;
    let mut num_batches = 0;

    for batch in samples.chunks(BATCH_SIZE) {
        let Batch { images, labels } =
            load_batch(batch, &eval_device, false).expect("failed to load validation batch");
        let logits = eval_model.forward(images);
        let predicted = logits.clone().argmax(1).reshape([batch.len()]);
        let matches: f32 = predicted.equal(labels.clone()).float().sum().into_scalar();

        correct += matches;
        total_loss += classification_loss(logits, labels).into_scalar::<f32>();
        num_batches += 1;
    }

    (
        total_loss / num_batches as f32,
        correct / samples.len() as f32,
    )
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
