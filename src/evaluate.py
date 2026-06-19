"""Evaluate a trained ResNet50 potato leaf disease classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from train import build_model, load_torch_state, make_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/PotatoPlants"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/best_model.pth"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, class_names = make_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = build_model(
        num_classes=len(class_names),
        device=device,
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(load_torch_state(args.checkpoint, device))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            running_loss += loss.item() * inputs.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    loss = running_loss / len(test_loader.dataset)
    print(f"Classes: {class_names}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
