use std::error::Error;
use std::path::Path;

use burn::tensor::{Device, Tensor, TensorData};
use image::imageops::FilterType;

pub const IMAGE_WIDTH: usize = 32;
pub const IMAGE_HEIGHT: usize = 32;
pub const CHANNELS: usize = 3;
pub const CLASS_NAMES: [&str; 2] = ["cats", "dogs"];

pub const TRAIN_DIRECTORY: &str = "data/train";
pub const VALIDATION_DIRECTORY: &str = "data/valid";

pub fn class_index(class_name: &str) -> Option<usize> {
    CLASS_NAMES.iter().position(|name| *name == class_name)
}

pub fn load_image(path: impl AsRef<Path>, device: &Device) -> Result<Tensor<3>, Box<dyn Error>> {
    let image = image::open(path)?
        .resize_exact(
            IMAGE_WIDTH as u32,
            IMAGE_HEIGHT as u32,
            FilterType::Triangle,
        )
        .to_rgb8();

    let mut pixels = vec![0.0; CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH];
    for (pixel_index, pixel) in image.pixels().enumerate() {
        let row = pixel_index / IMAGE_WIDTH;
        let column = pixel_index % IMAGE_WIDTH;

        for channel in 0..CHANNELS {
            let channel_offset = channel * IMAGE_HEIGHT * IMAGE_WIDTH;
            pixels[channel_offset + row * IMAGE_WIDTH + column] = f32::from(pixel[channel]) / 255.0;
        }
    }

    Ok(Tensor::from_data(
        TensorData::new(pixels, [CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH]),
        device,
    ))
}

#[cfg(test)]
mod tests {
    use image::{ImageBuffer, Rgb};

    use super::class_index;
    use super::{CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, load_image};

    #[test]
    fn class_names_map_to_stable_indices() {
        assert_eq!(class_index("cats"), Some(0));
        assert_eq!(class_index("dogs"), Some(1));
        assert_eq!(class_index("birds"), None);
    }

    #[test]
    fn images_become_normalized_channel_first_tensors() {
        let source = ImageBuffer::from_fn(2, 1, |x, _| {
            if x == 0 {
                Rgb([255u8, 0, 0])
            } else {
                Rgb([0u8, 128, 0])
            }
        });
        let path = std::env::temp_dir().join("vit-dataset-test.png");
        source.save(&path).unwrap();

        let device = Default::default();
        let tensor = load_image(&path, &device).unwrap();

        assert_eq!(tensor.dims(), [CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH]);
        assert!(
            tensor
                .to_data()
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .all(|value| { (0.0..=1.0).contains(value) })
        );

        std::fs::remove_file(path).unwrap();
    }
}
