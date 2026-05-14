from pathlib import Path

DATASETS_ROOT = Path("G:/TCC") 

oral_epithelium_db_name = "OralEpitheliumDB"
glas_db_name = "GlaS"
ocdc_db_name = "OCDC"
cryonuseg_db_name = "CryoNuSeg"

oral_epithelium_db_path = DATASETS_ROOT / "OralEpitheliumDB"
glas_db_path            = DATASETS_ROOT / "GlaS"
ocdc_db_path            = DATASETS_ROOT / "OCDC"
cryonuseg_db_path        = DATASETS_ROOT / "CryoNuSeg"


oral_epithelium_db_original = {
    "name": oral_epithelium_db_name,
    "variant": "original",
    "train": str(oral_epithelium_db_path / "original" / "train"),
    "val": str(oral_epithelium_db_path / "original" / "val"),
    "test": str(oral_epithelium_db_path / "original" / "test"),
}

glas_db_original = {
    "name": glas_db_name,
    "variant": "original",
    "train": str(glas_db_path / "original" / "train"),
    "val": str(glas_db_path / "original" / "val"),
    "test": str(glas_db_path / "original" / "test"),
}

ocdc_db_original = {
    "name": ocdc_db_name,
    "variant": "original",
    "train": str(ocdc_db_path / "original" / "train"),
    "val": str(ocdc_db_path / "original" / "val"),
    "test": str(ocdc_db_path / "original" / "test"),
}

cryonuseg_db_original = {
    "name": cryonuseg_db_name,
    "variant": "original",
    "train": str(cryonuseg_db_path / "original" / "train"),
    "val": str(cryonuseg_db_path / "original" / "val"),
    "test": str(cryonuseg_db_path / "original" / "test"),
}

oral_epithelium_db_augmented_rc_aug_f3 = {
    "name": oral_epithelium_db_name,
    "variant": "rc-aug-3",
    "train": str(oral_epithelium_db_path / "aug" / "rc-aug" / "rc-aug-3" / "train"),
    "val": str(oral_epithelium_db_path / "aug" / "rc-aug" / "rc-aug-3" / "val"),
    "test": str(oral_epithelium_db_path / "aug" / "rc-aug" / "rc-aug-3" / "test"),
}

glas_db_augmented_rc_aug_f3 = {
    "name": glas_db_name,
    "variant": "rc-aug-3",
    "train": str(glas_db_path / "aug" / "rc-aug" / "rc-aug-3" / "train"),
    "val": str(glas_db_path / "aug" / "rc-aug" / "rc-aug-3" / "val"),
    "test": str(glas_db_path / "aug" / "rc-aug" / "rc-aug-3" / "test"),
}

ocdc_db_augmented_rc_aug_f3 = {
    "name": ocdc_db_name,
    "variant": "rc-aug-3",
    "train": str(ocdc_db_path / "aug" / "rc-aug" / "rc-aug-3" / "train"),
    "val": str(ocdc_db_path / "aug" / "rc-aug" / "rc-aug-3" / "val"),
    "test": str(ocdc_db_path / "aug" / "rc-aug" / "rc-aug-3" / "test"),
}

cryonuseg_db_augmented_rc_aug_f3 = {
    "name": cryonuseg_db_name,
    "variant": "rc-aug-3",
    "train": str(cryonuseg_db_path / "aug" / "rc-aug" / "rc-aug-3" / "train"),
    "val": str(cryonuseg_db_path / "aug" / "rc-aug" / "rc-aug-3" / "val"),
    "test": str(cryonuseg_db_path / "aug" / "rc-aug" / "rc-aug-3" / "test"),
}

oral_epithelium_db_augmented_geometric_f3 = {
    "name": oral_epithelium_db_name,
    "variant": "geometric-3",
    "train": str(
        oral_epithelium_db_path / "aug" / "geometric" / "geometric-3" / "train"
    ),
    "val": str(oral_epithelium_db_path / "aug" / "geometric" / "geometric-3" / "val"),
    "test": str(oral_epithelium_db_path / "aug" / "geometric" / "geometric-3" / "test"),
}

glas_db_augmented_geometric_f3 = {
    "name": glas_db_name,
    "variant": "geometric-3",
    "train": str(glas_db_path / "aug" / "geometric" / "geometric-3" / "train"),
    "val": str(glas_db_path / "aug" / "geometric" / "geometric-3" / "val"),
    "test": str(glas_db_path / "aug" / "geometric" / "geometric-3" / "test"),
}

ocdc_db_augmented_geometric_f3 = {
    "name": ocdc_db_name,
    "variant": "geometric-3",
    "train": str(ocdc_db_path / "aug" / "geometric" / "geometric-3" / "train"),
    "val": str(ocdc_db_path / "aug" / "geometric" / "geometric-3" / "val"),
    "test": str(ocdc_db_path / "aug" / "geometric" / "geometric-3" / "test"),
}

cryonuseg_db_augmented_geometric_f3 = {
    "name": cryonuseg_db_name,
    "variant": "geometric-3",
    "train": str(cryonuseg_db_path / "aug" / "geometric" / "geometric-3" / "train"),
    "val": str(cryonuseg_db_path / "aug" / "geometric" / "geometric-3" / "val"),
    "test": str(cryonuseg_db_path / "aug" / "geometric" / "geometric-3" / "test"),
}

oral_epithelium_db_augmented_distortion_f3 = {
    "name": oral_epithelium_db_name,
    "variant": "distortion-3",
    "train": str(
        oral_epithelium_db_path / "aug" / "distortion" / "distortion-3" / "train"
    ),
    "val": str(oral_epithelium_db_path / "aug" / "distortion" / "distortion-3" / "val"),
    "test": str(
        oral_epithelium_db_path / "aug" / "distortion" / "distortion-3" / "test"
    ),
}

glas_db_augmented_distortion_f3 = {
    "name": glas_db_name,
    "variant": "distortion-3",
    "train": str(glas_db_path / "aug" / "distortion" / "distortion-3" / "train"),
    "val": str(glas_db_path / "aug" / "distortion" / "distortion-3" / "val"),
    "test": str(glas_db_path / "aug" / "distortion" / "distortion-3" / "test"),
}

ocdc_db_augmented_distortion_f3 = {
    "name": ocdc_db_name,
    "variant": "distortion-3",
    "train": str(ocdc_db_path / "aug" / "distortion" / "distortion-3" / "train"),
    "val": str(ocdc_db_path / "aug" / "distortion" / "distortion-3" / "val"),
    "test": str(ocdc_db_path / "aug" / "distortion" / "distortion-3" / "test"),
}

cryonuseg_db_augmented_distortion_f3 = {
    "name": cryonuseg_db_name,
    "variant": "distortion-3",
    "train": str(cryonuseg_db_path / "aug" / "distortion" / "distortion-3" / "train"),
    "val": str(cryonuseg_db_path / "aug" / "distortion" / "distortion-3" / "val"),
    "test": str(cryonuseg_db_path / "aug" / "distortion" / "distortion-3" / "test"),
}

oral_epithelium_db_augmented_cutout_f3 = {
    "name": oral_epithelium_db_name,
    "variant": "cutout-3",
    "train": str(oral_epithelium_db_path / "aug" / "cutout" / "cutout-3" / "train"),
    "val": str(oral_epithelium_db_path / "aug" / "cutout" / "cutout-3" / "val"),
    "test": str(oral_epithelium_db_path / "aug" / "cutout" / "cutout-3" / "test"),
}

glas_db_augmented_cutout_f3 = {
    "name": glas_db_name,
    "variant": "cutout-3",
    "train": str(glas_db_path / "aug" / "cutout" / "cutout-3" / "train"),
    "val": str(glas_db_path / "aug" / "cutout" / "cutout-3" / "val"),
    "test": str(glas_db_path / "aug" / "cutout" / "cutout-3" / "test"),
}

ocdc_db_augmented_cutout_f3 = {
    "name": ocdc_db_name,
    "variant": "cutout-3",
    "train": str(ocdc_db_path / "aug" / "cutout" / "cutout-3" / "train"),
    "val": str(ocdc_db_path / "aug" / "cutout" / "cutout-3" / "val"),
    "test": str(ocdc_db_path / "aug" / "cutout" / "cutout-3" / "test"),
}

cryonuseg_db_augmented_cutout_f3 = {
    "name": cryonuseg_db_name,
    "variant": "cutout-3",
    "train": str(cryonuseg_db_path / "aug" / "cutout" / "cutout-3" / "train"),
    "val": str(cryonuseg_db_path / "aug" / "cutout" / "cutout-3" / "val"),
    "test": str(cryonuseg_db_path / "aug" / "cutout" / "cutout-3" / "test"),
}

oral_epithelium_db_augmented_grid_dropout_f3 = {
    "name": oral_epithelium_db_name,
    "variant": "grid-dropout-3",
    "train": str(
        oral_epithelium_db_path / "aug" / "grid-dropout" / "grid-dropout-3" / "train"
    ),
    "val": str(
        oral_epithelium_db_path / "aug" / "grid-dropout" / "grid-dropout-3" / "val"
    ),
    "test": str(
        oral_epithelium_db_path / "aug" / "grid-dropout" / "grid-dropout-3" / "test"
    ),
}

glas_db_augmented_grid_dropout_f3 = {
    "name": glas_db_name,
    "variant": "grid-dropout-3",
    "train": str(glas_db_path / "aug" / "grid-dropout" / "grid-dropout-3" / "train"),
    "val": str(glas_db_path / "aug" / "grid-dropout" / "grid-dropout-3" / "val"),
    "test": str(glas_db_path / "aug" / "grid-dropout" / "grid-dropout-3" / "test"),
}

ocdc_db_augmented_grid_dropout_f3 = {
    "name": ocdc_db_name,
    "variant": "grid-dropout-3",
    "train": str(ocdc_db_path / "aug" / "grid-dropout" / "grid-dropout-3" / "train"),
    "val": str(ocdc_db_path / "aug" / "grid-dropout" / "grid-dropout-3" / "val"),
    "test": str(ocdc_db_path / "aug" / "grid-dropout" / "grid-dropout-3" / "test"),
}

cryonuseg_db_augmented_grid_dropout_f3 = {
    "name": cryonuseg_db_name,
    "variant": "grid-dropout-3",
    "train": str(
        cryonuseg_db_path / "aug" / "grid-dropout" / "grid-dropout-3" / "train"
    ),
    "val": str(cryonuseg_db_path / "aug" / "grid-dropout" / "grid-dropout-3" / "val"),
    "test": str(cryonuseg_db_path / "aug" / "grid-dropout" / "grid-dropout-3" / "test"),
}


all_data_sets = [
    oral_epithelium_db_original,
    oral_epithelium_db_augmented_rc_aug_f3,
    oral_epithelium_db_augmented_geometric_f3,
    oral_epithelium_db_augmented_distortion_f3,
    oral_epithelium_db_augmented_cutout_f3,
    oral_epithelium_db_augmented_grid_dropout_f3,

    glas_db_original,
    glas_db_augmented_rc_aug_f3,
    glas_db_augmented_geometric_f3,
    glas_db_augmented_distortion_f3,
    glas_db_augmented_cutout_f3,
    glas_db_augmented_grid_dropout_f3,

    ocdc_db_original,
    ocdc_db_augmented_rc_aug_f3,
    ocdc_db_augmented_geometric_f3,
    ocdc_db_augmented_distortion_f3,
    ocdc_db_augmented_cutout_f3,
    ocdc_db_augmented_grid_dropout_f3,

    cryonuseg_db_original,
    cryonuseg_db_augmented_rc_aug_f3,
    cryonuseg_db_augmented_geometric_f3,
    cryonuseg_db_augmented_distortion_f3,
    cryonuseg_db_augmented_cutout_f3,
    cryonuseg_db_augmented_grid_dropout_f3,
]


def has_images(directory: Path):
    valid_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    for sub in directory.iterdir():
        if sub.is_dir():
            for file in sub.iterdir():
                if file.suffix.lower() in valid_ext:
                    return True
    return False


def check_dataset_structure(dataset_list):
    for ds in dataset_list:
        print(f"Dataset: {ds['name']} | Variant: {ds['variant']}")
        for split in ["train", "val", "test"]:
            split_path = Path(ds[split])
            print(f"  Checking {split} at: {split_path}")
            if not split_path.is_dir():
                print(f"    {split} directory does not exist.")
                continue

            images_dir = split_path / "images"
            masks_dir = split_path / "masks"

            if not images_dir.is_dir():
                print("    'images/' subdirectory is missing.")
            else:
                if has_images(images_dir):
                    print("    'images/' OK")
                else:
                    print(
                        "    'images/' exists, but no images were found in subfolders."
                    )

            if not masks_dir.is_dir():
                print("    'masks/' subdirectory is missing.")
            else:
                if has_images(masks_dir):
                    print("    'masks/' OK")
                else:
                    print("    'masks/' exists, but no masks were found in subfolders.")
        print()
