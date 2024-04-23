import sys, torch
from common import BATCH_SIZE, DATA_DIR, TRANSFORMS, VAL_TRANSFORM
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchvision.transforms import v2
from tqdm import tqdm


def val(transform, model_name):
    """Evaluate the model with the given transformation."""

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.multiprocessing.set_sharing_strategy("file_system")

    # Load the data.
    transform = v2.Compose([transform, VAL_TRANSFORM])
    val_dataset = CIFAR10(DATA_DIR, train=False, download=True, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)

    # Define the model.
    model = resnet18()
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, 10),
    )
    model.load_state_dict(torch.load(f"models/{model_name}.pth"))
    model = model.to(device)

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
    assert len(sys.argv) == 2
    model_name = sys.argv[1]

    for transform in TRANSFORMS:
        val(transform, model_name)


if __name__ == "__main__":
    main()
