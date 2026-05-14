import shutil
import random
from pathlib import Path

# ─── CONFIGURAÇÕES ────────────────────────────────────────────────
BASE_DIR = Path("G:/TCC/OralEpitheliumDB")

IMAGES_DIR = BASE_DIR / "Original ROI images"
MASKS_DIR  = BASE_DIR / "Gold_Standard_Semantic_Segmentation"
OUTPUT_DIR = BASE_DIR / "original"

CLASSES = ["healthy", "severe"]

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

SEED = 42
# ──────────────────────────────────────────────────────────────────

random.seed(SEED)

def create_dirs():
    for split in ["train", "val", "test"]:
        for kind in ["images", "masks"]:
            for cls in CLASSES:
                Path(OUTPUT_DIR / split / kind / cls).mkdir(parents=True, exist_ok=True)

def split_files(files):
    random.shuffle(files)
    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    train = files[:n_train]
    val   = files[n_train:n_train + n_val]
    test  = files[n_train + n_val:]
    return train, val, test

def copy_files(files, cls, split):
    for fname in files:  # fname é só o nome do arquivo, ex: healthy-01-roi1.tif
        src_img = IMAGES_DIR / cls / fname
        dst_img = OUTPUT_DIR / split / "images" / cls / fname
        shutil.copy2(src_img, dst_img)

        mask_fname = Path(fname).stem + ".png"
        src_mask = MASKS_DIR / cls / mask_fname
        dst_mask = OUTPUT_DIR / split / "masks" / cls / mask_fname

        if src_mask.exists():
            shutil.copy2(src_mask, dst_mask)
        else:
            print(f"  ⚠️  Máscara não encontrada para: {fname}")

def main():
    create_dirs()

    for cls in CLASSES:
        img_folder = IMAGES_DIR / cls
        # .name aqui é a correção — salva só o nome do arquivo, não o caminho completo
        all_files = sorted([
            f.name for f in img_folder.iterdir()
            if f.suffix.lower() == ".tif"
        ])

        print(f"\nClasse '{cls}': {len(all_files)} imagens encontradas")

        train, val, test = split_files(all_files)

        print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

        copy_files(train, cls, "train")
        copy_files(val,   cls, "val")
        copy_files(test,  cls, "test")

    print("\n✅ Dataset organizado com sucesso em:", OUTPUT_DIR)

if __name__ == "__main__":
    main()