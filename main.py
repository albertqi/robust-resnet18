import torch
from common import DATA_DIR, TRANSFORMS
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageNet
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.transforms import v2
from tqdm import tqdm


NUM_EPOCHS = 50
TRAIN_SUBSET_SIZE = 0.05
VAL_SUBSET_SIZE = 0.25


class RandomTransform:
    """Randomly apply a transformation."""

    def __init__(self, preprocessing, transforms):
        self.preprocessing = preprocessing
        self.transforms = transforms

    def __call__(self, img):
        transform = self.transforms[
            torch.randint(len(self.transforms), size=(1,)).item()
        ]
        composed = (
            v2.Compose([self.preprocessing, transform])
            if transform
            else self.preprocessing
        )
        return composed(img)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Define the preprocessing pipeline.
    weights = ResNet34_Weights.DEFAULT
    preprocessing = weights.transforms()

    # Load the training data.
    transform = RandomTransform(preprocessing, TRANSFORMS)
    train_dataset = ImageNet(DATA_DIR, split="train", transform=transform)
    subset_size = int(TRAIN_SUBSET_SIZE * len(train_dataset))
    subset_indices = torch.randperm(len(train_dataset))[:subset_size].tolist()
    sampler = SubsetRandomSampler(subset_indices)
    train_dataloader = DataLoader(
        train_dataset, batch_size=32, num_workers=8, sampler=sampler
    )

    # Load the validation data.
    val_dataset = ImageNet(DATA_DIR, split="val", transform=transform)
    subset_size = int(VAL_SUBSET_SIZE * len(val_dataset))
    subset_indices = torch.randperm(len(val_dataset))[:subset_size].tolist()
    sampler = SubsetRandomSampler(subset_indices)
    val_dataloader = DataLoader(
        val_dataset, batch_size=32, num_workers=8, sampler=sampler
    )

    # Define the model.
    model = resnet34().to(device)

    # Define the optimizer.
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    # Define the loss function.
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        # Train the model.
        losses, accuracies = [], []
        model.train()
        for X, y in tqdm(train_dataloader):
            X, y = X.to(device), y.to(device)

            l = model(X)  # Forward pass.
            J = loss_fn(l, y)  # Compute the loss.
            optimizer.zero_grad()  # Zero the gradients.
            J.backward()  # Backward pass.
            optimizer.step()  # Update the parameters.

            losses.append(J.item())
            accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

        avg_loss = torch.tensor(losses).mean()
        avg_accuracy = torch.tensor(accuracies).mean()

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Avg. Training Loss: {avg_loss:.2f}")
        print(f"Avg. Training Accuracy: {avg_accuracy:.2f}")
        print()

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

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Avg. Validation Loss: {avg_loss:.2f}")
        print(f"Avg. Validation Accuracy: {avg_accuracy:.2f}")
        print()


if __name__ == "__main__":
    main()
