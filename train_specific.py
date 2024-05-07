import torch
from common import BATCH_SIZE, DATA_DIR, TRAIN_TRANSFORM, TRANSFORMS, VAL_TRANSFORM
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchvision.transforms import v2
from tqdm import tqdm


NUM_EPOCHS = 10


def generate_robust_model(transform):
    """Generate a model that is robust to the given transformation."""

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.multiprocessing.set_sharing_strategy("file_system")

    # Load the training data.
    train_transform = v2.Compose([transform, TRAIN_TRANSFORM])
    train_dataset = CIFAR10(DATA_DIR, train=True, download=True, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)

    # Load the validation data.
    val_transform = v2.Compose([transform, VAL_TRANSFORM])
    val_dataset = CIFAR10(DATA_DIR, train=False, download=True, transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)

    # Define the model.
    model = resnet18()
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, 10),
    )
    model.load_state_dict(torch.load("models/pretrained.pth"))
    model = model.to(device)

    # Define the optimizer.
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

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

    # Save the model.
    name = transform.__class__.__name__.lower()
    torch.save(model.state_dict(), f"models/{name}.pth")


def main():
    for transform in TRANSFORMS:
        if transform.__class__.__name__ == "Identity":
            continue
        generate_robust_model(transform)


if __name__ == "__main__":
    main()
