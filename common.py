from torchvision.transforms import v2


DATA_DIR = "data/"
BATCH_SIZE = 128
TRANSFORMS = [
    # No Transformation.
    v2.Identity(),

    # Geometric Transformations.
    v2.ElasticTransform(alpha=200.0, sigma=5.0),
    v2.RandomPerspective(distortion_scale=0.5, p=1.0),
    v2.RandomRotation(degrees=180.0),

    # Photometric Transformations.
    v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    v2.GaussianBlur(kernel_size=(9, 9), sigma=8.0),
    v2.RandomInvert(p=1.0),
    v2.RandomPosterize(bits=2, p=1.0),
]
TRAIN_TRANSFORM = v2.Compose([
    v2.RandomCrop(32, padding=4),
    v2.RandomHorizontalFlip(),
    v2.ToTensor(),
    v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
VAL_TRANSFORM = v2.Compose([
    v2.ToTensor(),
    v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
