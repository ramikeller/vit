mod dataset;
mod model;

fn main() {
    println!(
        "Dataset contract: {}x{}x{} images, classes {:?}, train={}, valid={}",
        dataset::IMAGE_WIDTH,
        dataset::IMAGE_HEIGHT,
        dataset::CHANNELS,
        dataset::CLASS_NAMES,
        dataset::TRAIN_DIRECTORY,
        dataset::VALIDATION_DIRECTORY
    );
}
