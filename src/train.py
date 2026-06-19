"""Train a ResNet50 classifier for potato leaf disease recognition."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


DEFAULT_DATA_DIR = Path("data/PotatoPlants")
DEFAULT_OUTPUT_DIR = Path("outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pretrained-weights", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def build_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def split_indices(
    total_size: int,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[list[int], list[int], list[int]]:
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(total_size, generator=generator).tolist()
    train_indices = shuffled[:train_size]
    val_indices = shuffled[train_size : train_size + val_size]
    test_indices = shuffled[train_size + val_size :]
    return train_indices, val_indices, test_indices


def make_dataloaders(
    data_dir: Path,
    batch_size: int,
    image_size: int,
    seed: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    train_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=build_train_transform(image_size),
    )
    eval_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=build_eval_transform(image_size),
    )

    train_indices, val_indices, test_indices = split_indices(
        total_size=len(train_dataset),
        seed=seed,
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        Subset(eval_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        Subset(eval_dataset, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader, train_dataset.classes


def load_torch_state(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(
    num_classes: int,
    device: torch.device,
    pretrained: bool = True,
    pretrained_weights: Path | None = None,
    freeze_backbone: bool = True,
) -> nn.Module:
    model = models.resnet50(weights=None)

    if pretrained:
        if pretrained_weights is not None:
            state_dict = load_torch_state(pretrained_weights, device)
            model.load_state_dict(state_dict)
        else:
            try:
                weights = models.ResNet50_Weights.DEFAULT
                model = models.resnet50(weights=weights)
            except Exception as exc:
                print(f"Could not load ImageNet weights automatically: {exc}")
                print("Continuing with randomly initialized weights.")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(loader.dataset), 100.0 * correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(loader.dataset), 100.0 * correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    output_dir: Path,
    epochs: int,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{epochs} "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pth")

    torch.save(model.state_dict(), output_dir / "last_model.pth")
    print(f"Done. Best validation accuracy: {best_acc:.2f}%")


def trainable_parameters(parameters: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    return [param for param in parameters if param.requires_grad]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, class_names = make_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    print(f"Classes: {class_names}")
    print(
        f"Train: {len(train_loader.dataset)}, "
        f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}"
    )

    model = build_model(
        num_classes=len(class_names),
        device=device,
        pretrained=not args.no_pretrained,
        pretrained_weights=args.pretrained_weights,
        freeze_backbone=True,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainable_parameters(model.parameters()), lr=args.lr)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        output_dir=args.output_dir,
        epochs=args.epochs,
        device=device,
    )


if __name__ == "__main__":
    main()
