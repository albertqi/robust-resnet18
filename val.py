import torch
from common import DATA_DIR, TRANSFORMS
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.transforms import v2
from tqdm import tqdm


def val(transform):
    """Evaluate the model with the given transformation."""

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Define the preprocessing pipeline.
    weights = ResNet34_Weights.DEFAULT
    preprocessing = weights.transforms()

    # Load the data.
    transform = v2.Compose([preprocessing, transform]) if transform else preprocessing
    val_dataset = ImageNet(DATA_DIR, split="val", transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=8)

    # Define the model.
    model = resnet34(weights=weights).to(device)

    # Define the loss function.
    loss_fn = nn.CrossEntropyLoss()

    # Evaluate the model.
    losses, accuracies = [], []
    model.eval()
    for X, y in tqdm(val_dataloader):
        X, y = X.to(device), y.to(device)

        with torch.no_grad():
            l = model(X)
        J = loss_fn(l, y)

        losses.append(J.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

    avg_loss = torch.tensor(losses).mean()
    avg_accuracy = torch.tensor(accuracies).mean()

    print(transform)
    print(f"Avg. Validation Loss: {avg_loss:.2f}")
    print(f"Avg. Validation Accuracy: {avg_accuracy:.2f}")
    print()


def main():
    for transform in TRANSFORMS:
        val(transform)


if __name__ == "__main__":
    main()
