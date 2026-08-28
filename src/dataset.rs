use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use burn::tensor::{Device, Int, Tensor, TensorData};
use image::imageops::FilterType;

pub const IMAGE_WIDTH: usize = 32;
pub const IMAGE_HEIGHT: usize = 32;
pub const CHANNELS: usize = 3;
pub const PATCH_SIZE: usize = 8;
pub const PATCHES_PER_SIDE: usize = IMAGE_WIDTH / PATCH_SIZE;
pub const NUM_PATCHES: usize = PATCHES_PER_SIDE * PATCHES_PER_SIDE;
pub const PATCH_VALUES: usize = CHANNELS * PATCH_SIZE * PATCH_SIZE;
pub const CLASS_NAMES: [&str; 2] = ["cats", "dogs"];

pub const TRAIN_DIRECTORY: &str = "data/train";
pub const VALIDATION_DIRECTORY: &str = "data/valid";

#[derive(Debug, PartialEq, Eq)]
pub struct Sample {
    pub path: PathBuf,
    pub label: usize,
}

pub struct Batch {
    pub images: Tensor<4>,
    pub labels: Tensor<1, Int>,
}

pub fn class_index(class_name: &str) -> Option<usize> {
    CLASS_NAMES.iter().position(|name| *name == class_name)
}

pub fn discover_samples(root: impl AsRef<Path>) -> Result<Vec<Sample>, Box<dyn Error>> {
    let mut class_directories = fs::read_dir(root.as_ref())?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|entry| entry.path().is_dir())
        .collect::<Vec<_>>();
    class_directories.sort_by_key(|entry| entry.file_name());

    let mut samples = Vec::new();
    for class_directory in class_directories {
        let class_name = class_directory.file_name();
        let class_name = class_name.to_string_lossy();
        let label = class_index(&class_name)
            .ok_or_else(|| format!("unknown class directory: {class_name}"))?;

        let mut image_paths = fs::read_dir(class_directory.path())?
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|entry| entry.path())
            .filter(|path| is_image_path(path))
            .collect::<Vec<_>>();
        image_paths.sort();

        samples.extend(image_paths.into_iter().map(|path| Sample { path, label }));
    }

    Ok(samples)
}

fn is_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png"
            )
        })
        .unwrap_or(false)
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

pub fn load_batch(samples: &[Sample], device: &Device) -> Result<Batch, Box<dyn Error>> {
    if samples.is_empty() {
        return Err("cannot load an empty batch".into());
    }

    let mut images = Vec::with_capacity(samples.len());
    let mut labels = Vec::with_capacity(samples.len());
    for sample in samples {
        images.push(load_image(&sample.path, device)?);
        labels.push(sample.label as i64);
    }

    Ok(Batch {
        images: Tensor::stack::<4>(images, 0),
        labels: Tensor::from_data(TensorData::new(labels, [samples.len()]), device),
    })
}

pub fn patchify(images: Tensor<4>) -> Tensor<3> {
    let [batch_size, channels, height, width] = images.dims();
    assert_eq!(channels, CHANNELS, "patchify expects RGB images");
    assert_eq!(
        height, IMAGE_HEIGHT,
        "patchify expects the configured image height"
    );
    assert_eq!(
        width, IMAGE_WIDTH,
        "patchify expects the configured image width"
    );

    images
        .reshape([
            batch_size,
            CHANNELS,
            PATCHES_PER_SIDE,
            PATCH_SIZE,
            PATCHES_PER_SIDE,
            PATCH_SIZE,
        ])
        .permute([0, 2, 4, 1, 3, 5])
        .reshape([batch_size, NUM_PATCHES, PATCH_VALUES])
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Tensor, TensorData};
    use image::{ImageBuffer, Rgb};

    use super::class_index;
    use super::{
        CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_PATCHES, PATCH_VALUES, Sample, discover_samples,
        load_batch, load_image, patchify,
    };

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

    #[test]
    fn samples_are_discovered_with_folder_labels() {
        let root = std::env::temp_dir().join(format!("vit-discovery-test-{}", std::process::id()));
        let cats = root.join("cats");
        let dogs = root.join("dogs");
        std::fs::create_dir_all(&cats).unwrap();
        std::fs::create_dir_all(&dogs).unwrap();
        std::fs::write(cats.join("second.txt"), b"ignored").unwrap();
        std::fs::write(cats.join("first.png"), b"image placeholder").unwrap();
        std::fs::write(dogs.join("dog.JPG"), b"image placeholder").unwrap();

        let samples = discover_samples(&root).unwrap();

        assert_eq!(
            samples,
            vec![
                Sample {
                    path: cats.join("first.png"),
                    label: 0,
                },
                Sample {
                    path: dogs.join("dog.JPG"),
                    label: 1,
                },
            ]
        );

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn samples_become_a_batch_with_aligned_labels() {
        let root = std::env::temp_dir().join(format!("vit-batch-test-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();

        let first_path = root.join("first.png");
        let second_path = root.join("second.png");
        ImageBuffer::from_pixel(2, 2, Rgb([255u8, 0, 0]))
            .save(&first_path)
            .unwrap();
        ImageBuffer::from_pixel(2, 2, Rgb([0u8, 255, 0]))
            .save(&second_path)
            .unwrap();

        let samples = vec![
            Sample {
                path: first_path,
                label: 0,
            },
            Sample {
                path: second_path,
                label: 1,
            },
        ];
        let device = Default::default();
        let batch = load_batch(&samples, &device).unwrap();

        assert_eq!(
            batch.images.dims(),
            [2, CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH]
        );
        assert_eq!(batch.labels.dims(), [2]);
        assert_eq!(batch.labels.to_data().as_slice::<i64>().unwrap(), &[0, 1]);

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn image_batches_become_patch_sequences() {
        let device = Default::default();
        let images = Tensor::from_data(
            TensorData::new(
                vec![0.0; 2 * CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH],
                [2, CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH],
            ),
            &device,
        );

        let patches = patchify(images);

        assert_eq!(patches.dims(), [2, NUM_PATCHES, PATCH_VALUES]);
    }
}
