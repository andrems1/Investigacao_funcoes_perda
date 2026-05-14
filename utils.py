import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_curves(experiment_dir, experiment_name, history):
    # Loss curve
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - {experiment_name}")
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, "loss_curve.png"))
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curve - {experiment_name}")
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, "accuracy_curve.png"))
    plt.close()

def save_hyperparameters(
    experiment_dir,
    experiment_name,
    experiment_model,
    dataset_path,
    image_size,
    seed,
    batch_size,
    learning_rate,
    num_epochs,
    use_scheduler,
    scheduler_params,
    use_early_stopping,
    patience,
    dropout_rate,
    batchnorm,
    timestamp,
    train_percent=0.6,
    val_percent=0.2,
    test_percent=0.2,
):
    hyperparameters = {
        "experiment_name": experiment_name,
        "model": experiment_model,
        "dataset_path": dataset_path,
        "image_size": list(image_size),
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "use_scheduler": use_scheduler,
        "scheduler_params": scheduler_params if use_scheduler else None,
        "use_early_stopping": use_early_stopping,
        "patience": patience if use_early_stopping else None,
        "train_percent": train_percent,
        "val_percent": val_percent,
        "test_percent": test_percent,
        "dropout_rate": dropout_rate,
        "batchnorm": batchnorm,
        "timestamp": timestamp,
    }

    with open(os.path.join(experiment_dir, "hyperparameters.yml"), "w") as f:
        yaml.dump(hyperparameters, f, sort_keys=False, default_flow_style=False)

    print(f"Hyperparameters saved at {experiment_dir}/hyperparameters.yml")
