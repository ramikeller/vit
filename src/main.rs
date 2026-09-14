mod dataset;
mod model;
mod training;

use burn::tensor::Device;
use rand::seq::SliceRandom;

use dataset::{TRAIN_DIRECTORY, VALIDATION_DIRECTORY, discover_samples};
use training::{evaluate, new_adam, train_epoch};

const NUM_EPOCHS: usize = 5;

fn main() {
    let device = Device::default().autodiff();
    let mut train_samples =
        discover_samples(TRAIN_DIRECTORY).expect("failed to discover training samples");
    let valid_samples =
        discover_samples(VALIDATION_DIRECTORY).expect("failed to discover validation samples");

    let mut model = model::VisionTransformer::new(&device);
    let mut optimizer = new_adam();
    let mut rng = rand::rng();

    for epoch in 1..=NUM_EPOCHS {
        train_samples.shuffle(&mut rng);

        let (updated_model, train_loss) =
            train_epoch(model, &mut optimizer, &train_samples, &device);
        model = updated_model;

        let (valid_loss, valid_accuracy) = evaluate(&model, &valid_samples, &device);

        println!(
            "epoch {epoch}/{NUM_EPOCHS}: train_loss={train_loss:.4} valid_loss={valid_loss:.4} valid_accuracy={:.1}%",
            valid_accuracy * 100.0
        );
    }
}
