# U-Net-Based Segmentation Experiments (Histology Datasets)

This repository contains the PyTorch code used to train and evaluate several U-Net-based architectures on multiple histology segmentation datasets (CryoNuSeg, GlaS, OralEpitheliumDB, OCDC), including different data augmentation variants (original, geometric, distortion, Cutout, Grid Dropout, RCAug), as described in the paper: 

**A U-Net-Based Approach for Histological Tissue Segmentation Using RCAug Data Augmentation**  
- Paper: https://ieeexplore.ieee.org/document/11223478
- Segmented masks and results: https://zenodo.org/records/17754089

---

Models implemented:

- `UNet`
- `UNetPP`
- `SharpUNet`
- `AttentionUNet`
- `ResidualUNet`
- `TransUNet`

---

Main scripts:

- `main.py` – builds experiments and runs training + testing
- `segmentation_dataset.py` – dataset abstraction
- `architectures.py` – model definitions
- `models.py` – model factory (`build_model`)
- `train.py` – training loop and evaluation
- `loss.py` – loss function
- `datasets.py` – paths and metadata for each dataset/variant

---

The repository assumes that images and masks are already preprocessed. No data augmentation is performed internally, so augmentation must be handled externally before training.

## 1. Requirements

Recommended:

- Python 3.9–3.11
- PyTorch (with CUDA if you have GPU)
- torchvision
- numpy
- pandas
- matplotlib
- pillow
- opencv-python
- scikit-image
- tqdm
- einops

Example installation:

```bash
pip install torch torchvision # use the correct index-url for your CUDA/CPU setup
pip install numpy pandas matplotlib pillow opencv-python scikit-image tqdm einops
```

## 2. Datasets

### 2.1. Download Datasets
You must download the datasets manually from their original sources:
- CryoNuSeg: https://github.com/masih4/CryoNuSeg
- GlaS: https://www.kaggle.com/datasets/livepriyanka09/warwick-gland-dataset
- OralEpitheliumDB: https://zenodo.org/records/17693395
- OCDC: https://data.mendeley.com/datasets/9bsc36jyrt/1

### 2.2. Dataset Structure

Each dataset (and each augmentation variant) must follow the folder structure below.  
This structure is the same for **train**, **val**, and **test** roots defined in `datasets.py`:

```text
<dataset_root>/
  images/
    <class_1>/
      image_001.tif
      image_002.png
      ...
    <class_2>/
      ...
  masks/
    <class_1>/
      image_001.png
      image_002.png
      ...
    <class_2>/
      ...
```
Where:

- <dataset_root> is one of the paths passed in datasets.py ("train", "val", "test").
- <class_*> is the class folder name (e.g. no-class, low-grade, high-grade, etc.).
- The image and mask filenames must match by basename (filename without extension).
- Example:
   - images/no-class/sample_001.tif
   - masks/no-class/sample_001.png are treated as a (image, mask) pair.

Supported image/mask extensions: .png, .jpg, .jpeg, .tif, .tiff, .bmp.

### 2.3. Data Augmentation

The repository does not perform data augmentation online. All augmentation strategies (geometric, distortion, Cutout, Grid Dropout, RCAug) must be implemented externally, for example using an image augmentation library (e.g., Albumentations) or custom scripts.

- Geometric augmentations: rotations, flips.
- Distortion augmentations: elastic deformations, grid distortions, optical distortions.
- Cutout: random erasing of rectangular regions.
- Grid Dropout: dropping out grid-like regions.
- RCAug: random combinations of augmentations described in the [RCAug paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10406800/).


## 3. Running Experiments

All experiment configurations are defined at the bottom of `main.py`.

Example (already in the code):

```python
REPETITION = 1
BATCH_SIZE = 8
LEARNING_RATE = 0.0005
DROPOUT_RATE = 0.2
NUM_EPOCHS = 150

datasets = [
    cryonuseg_db_original,
    glas_db_original,
    oral_epithelium_db_original,
    ocdc_db_original,
]

models = [
    "UNet",
    "UNetPP",
    "SharpUNet",
    "AttentionUNet",
    "ResidualUNet",
    "TransUNet",
]

use_scheduler = False
```

Each `dataset` entry is defined in `datasets.py` and contains:

```python
{
    "name": "GlaS",
    "variant": "original",
    "train": "/path/to/train",
    "val":   "/path/to/val",
    "test":  "/path/to/test",
}
```

To launch all experiments defined in `main.py`:

```bash
python main.py
```

For each experiment, the code will:

1. Create an output folder under `experiments/` with a timestamp.
2. Save hyperparameters and training curves.
3. Train the selected model on the selected dataset.
4. Evaluate on the test set.
5. Save:
   - Per-image metrics (`segmentation_metrics.csv`)
   - Per-class metrics (`segmentation_metrics_class.csv`)
   - Global summary (`metrics_summary.txt`)
   - Inference time statistics (`inference_times.txt`)
   - Predicted masks under `segmentation_mask/<class>/`.

---


## 4. Citation

If you use this code or the experimental setup in your work, please cite the corresponding SIBGRAPI paper:

BibTex:
```bibtex
@INPROCEEDINGS{11223478,
  author={de Oliveira, Domingos L. L. and Tosta, Thaína A. A. and Neves, Leandro A. and Silva, Adriano B. and Martins, Alessandro S. and de Faria, Paulo R. and do Nascimento, Marcelo Z.},
  booktitle={2025 38th SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI)}, 
  title={A U-Net-Based Approach for Histological Tissue Segmentation Using RCAug Data Augmentation}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Training;Image segmentation;Accuracy;Pipelines;Computer architecture;Data augmentation;Transformers;Robustness;Standards;Cancer},
  doi={10.1109/SIBGRAPI67909.2025.11223478}}
```

Plain Text:
```
D. L. L. de Oliveira et al., "A U-Net-Based Approach for Histological Tissue Segmentation Using RCAug Data Augmentation," 2025 38th SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI), Salvador, Brazil, 2025, pp. 1-6, doi: 10.1109/SIBGRAPI67909.2025.11223478.
```
