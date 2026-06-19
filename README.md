# Potato Leaf Disease Classification

Computer vision project for classifying potato leaf images into three classes:
early blight, late blight, and healthy. The project uses transfer learning with
ResNet50 and includes Grad-CAM visualizations to inspect model attention on
misclassified leaves.

## Project Highlights

- Built a three-class potato leaf disease classifier using PyTorch and ResNet50.
- Used ImageNet transfer learning with a frozen convolutional backbone and a
  task-specific fully connected classification head.
- Applied image augmentation and normalization for training.
- Evaluated model performance on a held-out test split.
- Used Grad-CAM to visualize regions that influenced model predictions.

## Dataset

The project uses a potato leaf disease dataset from Kaggle/PlantVillage. The
local dataset used for this project contains:

| Class | Images |
| --- | ---: |
| `Potato___Early_blight` | 1,000 |
| `Potato___Late_blight` | 1,000 |
| `Potato___healthy` | 152 |

Raw image data is not committed to this repository. To reproduce the project,
place the dataset in:

```text
data/PotatoPlants/
  Potato___Early_blight/
  Potato___Late_blight/
  Potato___healthy/
```

## Method

The pipeline uses an 80/10/10 train-validation-test split with a fixed random
seed. Training images are randomly resized, cropped, horizontally flipped, and
normalized with ImageNet statistics. Validation and test images use deterministic
resize and center crop transforms.

The model is a ResNet50 classifier:

1. Load ResNet50 with ImageNet weights when available.
2. Freeze the convolutional backbone.
3. Replace the final fully connected layer with a 3-class classification head.
4. Train the classification head with cross-entropy loss and Adam.
5. Save the best validation checkpoint locally under `outputs/`.

Model checkpoints are intentionally excluded from GitHub because they are large
binary artifacts.

## Results

The cleaned notebook snapshot reports a test accuracy of **98.15%**. The
original homework report mentions **95.38%**; this difference is likely due to
checkpoint selection, stochastic augmentation, and rerun state. The code uses a
fixed split seed, but exact results can still vary across environments.

### Accuracy

![Training and validation accuracy](assets/accuracy.png)

### Loss

![Training and validation loss](assets/loss.png)

### Grad-CAM Misclassification Example

![Grad-CAM misclassification example](assets/gradcam_misclassified_1.png)

## Repository Structure

```text
.
|-- assets/
|   |-- accuracy.png
|   |-- loss.png
|   |-- gradcam_misclassified_1.png
|   |-- gradcam_misclassified_2.png
|   `-- gradcam_misclassified_3.png
|-- data/
|   `-- .gitkeep
|-- docs/
|   `-- original_report.docx
|-- notebooks/
|   `-- potato_leaf_disease_classification.ipynb
|-- src/
|   |-- train.py
|   |-- evaluate.py
|   `-- gradcam.py
|-- .gitignore
|-- LICENSE
|-- README.md
`-- requirements.txt
```

## Setup

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python src/train.py --data-dir data/PotatoPlants --epochs 20
```

Evaluate the best checkpoint:

```bash
python src/evaluate.py --data-dir data/PotatoPlants --checkpoint outputs/best_model.pth
```

Generate a Grad-CAM overlay:

```bash
python src/gradcam.py \
  --image data/PotatoPlants/Potato___Late_blight/example.jpg \
  --checkpoint outputs/best_model.pth \
  --output assets/example_gradcam.png
```

## Notes and Limitations

- The healthy class is much smaller than the disease classes, so accuracy alone
  is not enough to fully characterize model performance.
- Future improvements should include a confusion matrix, per-class precision,
  recall, F1 score, and stronger validation on external images.
- Raw data, model checkpoints, and archived homework files are excluded to keep
  this repository lightweight and suitable for GitHub.

## Original Report

The original course report is included at `docs/original_report.docx`.
