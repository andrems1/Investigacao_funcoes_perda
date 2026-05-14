from datetime import datetime
import os
import time
import torch
import torch.optim as optim
import gc
from torch.utils.data import DataLoader
from torchvision import transforms
from segmentation_dataset import SegmentationDataset
from models import build_model
from loss import criterion, criterion_protoseg
from train import train_model, test_and_save_results
from utils import set_seed, get_device, save_hyperparameters, plot_curves
from datasets import (
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
)


def build_transforms(image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    return transform, mask_transform


def create_datasets(dataset_path, dataset_val_path, dataset_test_path,
                    transform, mask_transform):
    train_dataset = SegmentationDataset(
        root_dir=dataset_path, transform=transform, mask_transform=mask_transform)
    val_dataset = SegmentationDataset(
        root_dir=dataset_val_path, transform=transform, mask_transform=mask_transform)
    test_dataset = SegmentationDataset(
        root_dir=dataset_test_path, transform=transform, mask_transform=mask_transform)
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader


def run_experiment(
    experiment_name,
    experiment_model,
    batch_size,
    learning_rate,
    num_epochs,
    image_size,
    use_scheduler,
    dropout_rate,
    batchnorm,
    dataset_path,
    dataset_val_path,
    dataset_test_path,
):
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    experiment_dir = f"experiments/{experiment_name}-{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)

    segmentation_mask_dir = os.path.join(experiment_dir, "segmentation_mask")
    os.makedirs(segmentation_mask_dir, exist_ok=True)

    seed = 42
    set_seed(seed)
    device = get_device()

    print(f"Using device: {device}")
    print(f"Experiment: {experiment_name} | Model: {experiment_model}")

    # ✅ Define qual criterion e se aplica sigmoid no teste
    # UNetProtoSeg já retorna probabilidades via softmax
    is_protoseg = (experiment_model == "UNetProtoSeg")
    loss_fn     = criterion_protoseg if is_protoseg else criterion
    apply_sig   = not is_protoseg

    transform, mask_transform = build_transforms(image_size)
    train_dataset, val_dataset, test_dataset = create_datasets(
        dataset_path, dataset_val_path, dataset_test_path, transform, mask_transform)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size)

    model = build_model(experiment_model, dropout_rate, batchnorm, device)

    scheduler_params = {
        "mode": "min", "factor": 0.5, "patience": 5,
        "cooldown": 3, "min_lr": 1e-5,
    }
    use_early_stopping = False
    patience = 10

    save_hyperparameters(
        experiment_dir=experiment_dir,
        experiment_name=experiment_name,
        experiment_model=experiment_model,
        dataset_path=dataset_path,
        image_size=image_size,
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        use_scheduler=use_scheduler,
        scheduler_params=scheduler_params,
        use_early_stopping=use_early_stopping,
        patience=patience,
        dropout_rate=dropout_rate,
        batchnorm=batchnorm,
        timestamp=timestamp,
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_params["mode"],
            factor=scheduler_params["factor"],
            patience=scheduler_params["patience"],
            cooldown=scheduler_params["cooldown"],
            min_lr=scheduler_params["min_lr"],
        )

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=loss_fn,        # ✅ criterion correto para cada modelo
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
        experiment_dir=experiment_dir,
        use_scheduler=use_scheduler,
        use_early_stopping=use_early_stopping,
        patience=patience,
    )

    plot_curves(experiment_dir, experiment_name, history)

    test_and_save_results(
        model=model,
        test_loader=test_loader,
        test_dataset=test_dataset,
        device=device,
        experiment_dir=experiment_dir,
        segmentation_mask_dir=segmentation_mask_dir,
        apply_sigmoid=apply_sig,  # ✅ False para ProtoSeg, True para os demais
    )


if __name__ == "__main__":
    REPETITION    = 1
    BATCH_SIZE    = 8
    LEARNING_RATE = 0.0005
    DROPOUT_RATE  = 0.2
    NUM_EPOCHS    = 150

    datasets = [
        oral_epithelium_db_original,
    ]

    models = [
        "UNetProtoSeg",  # <- rode primeiro para validar a integração
              # <- depois troque para a baseline sem ProtoSeg
    ]

    use_scheduler = False
    experiments   = []

    for repeat in range(REPETITION):
        for model_name in models:
            for dataset in datasets:
                image_size = (224, 224)
                for learning_rate in [LEARNING_RATE]:
                    for dropout_rate in [DROPOUT_RATE]:
                        for num_epochs in [NUM_EPOCHS]:
                            experiments.append({
                                "experiment_name": f"{dataset['name']}-{dataset['variant']}-{model_name}",
                                "experiment_model": model_name,
                                "batch_size":       BATCH_SIZE,
                                "learning_rate":    learning_rate,
                                "num_epochs":       num_epochs,
                                "image_size":       image_size,
                                "use_scheduler":    use_scheduler,
                                "dropout_rate":     dropout_rate,
                                "batchnorm":        True,
                                "dataset_path":     dataset["train"],
                                "dataset_val_path": dataset["val"],
                                "dataset_test_path":dataset["test"],
                            })

    for i, experiment in enumerate(experiments, start=1):
        print(f"Starting experiment {i}/{len(experiments)}: {experiment['experiment_name']}")
        print("Clearing GPU cache and collecting garbage...\n")
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.ipc_collect()

        run_experiment(**experiment)

        print("Experiment finished!")

        cooldown_minutes = 0.0
        cooldown_time = cooldown_minutes * 60
        if cooldown_time > 0:
            print(f"Waiting {cooldown_minutes} minutes before next experiment...\n")
            time.sleep(cooldown_time)