import matplotlib.pyplot as plt
import sys, torch
from common import DATA_DIR, TRANSFORMS
from torchvision.datasets import ImageNet


TRANFORM = None  # Transformation to visualize.


def visualize_transform(transform):
    """Visualize the transformation."""

    # Load the data.
    val_dataset = ImageNet(DATA_DIR, split="val", transform=transform)

    # Visualize the data.
    figure = plt.figure(figsize=(6, 6))
    cols, rows = 2, 2
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(val_dataset), size=(1,)).item()
        img, label = val_dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(f"Class {label}")
        plt.axis("off")
        plt.imshow(img)
    plt.show()


def visualize_all():
    """Visualize all transformations."""

    figure = plt.figure(figsize=(6, 12))
    cols, rows = 2, 4
    for i in range(1, cols * rows + 1):
        val_dataset = ImageNet(DATA_DIR, split="val", transform=TRANSFORMS[i - 1])
        sample_idx = torch.randint(len(val_dataset), size=(1,)).item()
        img, _ = val_dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(
            f"{TRANSFORMS[i - 1].__class__.__name__ if TRANSFORMS[i - 1] else 'None'}"
        )
        plt.axis("off")
        plt.imshow(img)
    plt.show()


def main():
    assert len(sys.argv) == 2
    should_visualize_all = sys.argv[1] == "1"

    if should_visualize_all:
        visualize_all()
    else:
        visualize_transform(TRANFORM)


if __name__ == "__main__":
    main()
