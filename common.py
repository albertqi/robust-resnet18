from torchvision.transforms import v2


DATA_DIR = "data/"
TRANSFORMS = [
    None,
    # Geometric transformations.
    v2.ElasticTransform(alpha=200.0, sigma=5.0),
    v2.RandomPerspective(distortion_scale=0.5, p=1.0),
    v2.RandomRotation(degrees=180.0),
    # Photometric transformations.
    v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    v2.GaussianBlur(kernel_size=(9, 9), sigma=8.0),
    v2.RandomInvert(p=1.0),
    v2.RandomPosterize(bits=2, p=1.0),
]
