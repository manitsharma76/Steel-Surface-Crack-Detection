# Steel Surface Defect Detection

Deep-learning project for binary segmentation of defects on steel surfaces. The complete workflow is in [Steel-surface-detection.ipynb](Steel-surface-detection.ipynb).

## Overview

- Decodes Severstal Steel Defect Detection run-length encoded masks.
- Trains a ResNet34 U-Net to predict defective pixels.
- Evaluates Dice, accuracy, precision, recall, F1, and confusion matrix metrics.
- Produces prediction overlays, training plots, and Grad-CAM explanations.
- Includes a reusable `predict_single_image` inference function.

## Dataset

Download the [Severstal Steel Defect Detection dataset](https://www.kaggle.com/competitions/severstal-steel-defect-detection):

```text
severstal-steel-defect-detection/
├── train.csv
├── train_images/
└── test_images/
```

Set the dataset location in the notebook:

```python
DATA_PATH = "/kaggle/input/competitions/severstal-steel-defect-detection"
```

All source defect types are merged into one foreground class: `0` is background and `1` is defect.

## Model

The main model is a U-Net with a pretrained ResNet34 encoder. Images and masks are resized to `256 x 256`; training uses paired augmentation, a weighted BCE plus Dice loss, and an 80/20 stratified train-validation split.

Training includes learning-rate reduction, early stopping, mixed precision on CUDA, and gradient clipping. The best checkpoint is saved as `resnet.pth`.

## Setup and Usage

Install the notebook dependencies:

```bash
pip install numpy pandas matplotlib opencv-python torch torchvision scikit-learn tqdm scipy albumentations seaborn jupyter
```

Open the notebook in Jupyter or VS Code, update `DATA_PATH`, and run the cells from top to bottom. CUDA is used when available; otherwise the notebook falls back to CPU.

For single-image inference after training or loading `resnet`:

```python
mask, probability_map, defect_ratio = predict_single_image(
    "path/to/image.jpg",
    resnet,
    threshold=0.5,
)
```

To load a saved checkpoint:

```python
resnet = ResNetUNet()
resnet.load_state_dict(
    torch.load("resnet.pth", map_location=DEVICE, weights_only=True)
)
resnet = resnet.to(DEVICE)
resnet.eval()
```

## Outputs

Generated visualizations are stored in [`images/`](images/): dataset distributions, sample masks, training curves, prediction grids, overlays, validation metrics, confusion matrices, Grad-CAM, and a summary dashboard.

## Limitations

- The four source defect types are merged into one binary class.
- Wide source images are resized directly to a square, which may distort geometry.
- Only one validation split is used.
- The `0.5` prediction threshold is not calibrated.
- The notebook does not include a production API or mask export pipeline.

## Project Structure

```text
├── Steel-surface-detection.ipynb   # Data preparation, training, evaluation, and inference
├── README.md                        # Project documentation
└── images/                          # Saved visualization outputs
```

## Visualization Gallery

### Dataset exploration

| Defect presence                                                                  | Defect classes                                                     |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| ![Defect versus no-defect distribution](images/defect_presence_distribution.png) | ![Defect class distribution](images/defect_class_distribution.png) |

| Sample images and masks                                                     | Mask pixel distribution                                        |
| --------------------------------------------------------------------------- | -------------------------------------------------------------- |
| ![Sample images and ground-truth masks](images/sample_images_and_masks.png) | ![Mask pixel distribution](images/mask_pixel_distribution.png) |

### Training and evaluation

![Training and validation curves](images/train_val_curves.png)

![Training history](images/training_history.png)

| Validation metrics                                   | Confusion matrix                                             |
| ---------------------------------------------------- | ------------------------------------------------------------ |
| ![Validation metrics](images/validation_metrics.png) | ![Pixel-level confusion matrix](images/confusion_matrix.png) |

### Predictions and explainability

![Prediction versus ground truth grid](images/pred_vs_gt_grid.png)

![Prediction overlay](images/prediction_overlay.png)

![Grad-CAM visualization](images/gradcam.png)

![Project summary dashboard](images/dashboard.png)
