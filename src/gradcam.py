"""Generate a Grad-CAM overlay for a trained potato leaf disease classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train import build_eval_transform, build_model, load_torch_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/best_model.pth"))
    parser.add_argument("--output", type=Path, default=Path("assets/example_gradcam.png"))
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--class-index", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = cam - cam.min()
    denominator = cam.max()
    if denominator > 0:
        cam = cam / denominator
    return cam


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        num_classes=args.num_classes,
        device=device,
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(load_torch_state(args.checkpoint, device))
    model.eval()

    activations = None
    gradients = None

    def forward_hook(_module, _inputs, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(_module, _grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    target_layer = model.layer4[-1].conv3
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    image = Image.open(args.image).convert("RGB")
    transform = build_eval_transform(args.image_size)
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.zero_grad()
    output = model(input_tensor)
    class_index = args.class_index
    if class_index is None:
        class_index = output.argmax(dim=1).item()
    output[0, class_index].backward()

    if activations is None or gradients is None:
        raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(
        cam,
        size=(args.image_size, args.image_size),
        mode="bilinear",
        align_corners=False,
    )
    cam_array = cam.squeeze().cpu().numpy()
    cam_array = normalize_cam(cam_array)

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_array),
        cv2.COLORMAP_JET,
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    base_image = np.array(image.resize((args.image_size, args.image_size)))
    overlay = (0.5 * base_image + 0.5 * heatmap).astype(np.uint8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(args.output)
    forward_handle.remove()
    backward_handle.remove()
    print(f"Predicted class index: {class_index}")
    print(f"Saved Grad-CAM overlay to {args.output}")


if __name__ == "__main__":
    main()
