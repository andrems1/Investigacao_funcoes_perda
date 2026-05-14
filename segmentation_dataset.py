import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image


class SegmentationDataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        mask_transform=None,
        binarize_mask=True,
    ):
        """
        Generic segmentation dataset.

        Expected folder structure:
            root_dir/
              images/<class>/*.png|jpg|tif|...
              masks/<class>/*.png|jpg|tif|...

        Matching is done by BASE NAME (without extension).
        Ex:
            images/no-class/Human_AdrenalGland_01.tif
            masks/no-class/Human_AdrenalGland_01.png
        will be matched as the same pair.
        """
        self.root_dir = str(root_dir)
        self.image_dir = os.path.join(self.root_dir, "images")
        self.mask_dir = os.path.join(self.root_dir, "masks")
        self.transform = transform
        self.mask_transform = mask_transform
        self.binarize_mask = binarize_mask

        self.image_paths = []
        self.mask_paths = []
        self.image_names = []

        # Only folders (ignore loose files)
        self.classes = sorted(
            [
                d
                for d in os.listdir(self.image_dir)
                if os.path.isdir(os.path.join(self.image_dir, d))
            ]
        )

        # accepted extensions
        img_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        mask_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

        for class_name in self.classes:
            img_folder = os.path.join(self.image_dir, class_name)
            mask_folder = os.path.join(self.mask_dir, class_name)

            if not os.path.isdir(mask_folder):
                raise ValueError(
                    f"Mask folder not found for class '{class_name}': {mask_folder}"
                )

            img_files = sorted(
                [
                    f
                    for f in os.listdir(img_folder)
                    if f.lower().endswith(img_exts)
                ]
            )
            mask_files = sorted(
                [
                    f
                    for f in os.listdir(mask_folder)
                    if f.lower().endswith(mask_exts)
                ]
            )

            # map base name -> file name    
            img_map = {Path(f).stem: f for f in img_files}
            mask_map = {Path(f).stem: f for f in mask_files}

            common_basenames = sorted(set(img_map.keys()) & set(mask_map.keys()))
            missing_in_masks = sorted(set(img_map.keys()) - set(mask_map.keys()))
            missing_in_imgs = sorted(set(mask_map.keys()) - set(img_map.keys()))

            

            if missing_in_masks:
                print(
                    f"[WARN] Classe '{class_name}' em '{root_dir}': "
                    f"sem máscara para: {missing_in_masks}"
                )
            if missing_in_imgs:
                print(
                    f"[WARN] Classe '{class_name}' em '{root_dir}': "
                    f"sem imagem para: {missing_in_imgs}"
                )

            if not common_basenames:
                raise ValueError(
                    f"No matching image/mask pairs found for class '{class_name}' "
                    f"in '{root_dir}'. Check filenames."
                )

            for base in common_basenames:
                img = img_map[base]
                mask = mask_map[base]

                img_path = os.path.join(img_folder, img)
                mask_path = os.path.join(mask_folder, mask)

                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
                self.image_names.append(base)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image_name = self.image_names[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.binarize_mask:
            # binariza: 0 ou 255
            mask = mask.point(lambda p: 255 if p > 0 else 0)

        if self.transform is not None:
            image = self.transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        class_name = os.path.basename(os.path.dirname(img_path))
        label = self.classes.index(class_name)

        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(label, dtype=torch.long),
            "image_name": image_name,
            "class": class_name,
        }

    def get_classes(self):
        return self.classes
